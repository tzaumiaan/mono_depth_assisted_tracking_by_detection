"""Inference core module.

This module is the top level wrapper for the whole detection-by-tracking
framework. The wrapper takes care of a set of building blocks, including
detector, estimator, DA solver, tracker, and a bunch of unitilies to 
provide a clean interface to be executed frame by frame.

"""
# package dependency
import numpy as np
from absl import logging
from datetime import datetime

# internal dependency
from detector import detector
from estimator import estimator
from tracker import tracker
from da_solver import associate
from geo_utils import egomo_vec2mat, cam_proj, coord_transform
from common_utils import get_tdiff

class inference_wrapper:
  """Inference wrapper class.
  
  This class works as the wrapper of the inference core. It is initialized to 
  set up detector and estimator instances, and then it provides `run_frame`
  to be used directly frame by frame. Within it there are 2 major functions:
  `main_pipeline` and `traj_pipeline`, taking care of 2D plane detection and
  tracking and 3D trjectory projection, respectively.
  
  attributes:
    det (detector): Object detector instance.
    est (estimator): Depth and egomotion estimator instance.
    t_list (list of tracker): List of trackers.
    tid_new (int): Global latest index for new tracker. A unique tracker ID is
                   issued to a newly created tracker based on this global index
                   to prevent ID conflict.
    image_seq (list of numpy.array): List of a triplet of images, stored for 
                                     egomotion detection.
    egomo_trmat (numpy.array): Egomotion transformation matrix, stored as the 
                               accumulated ego pose from frame 0, dim = (4, 4).
    frame_idx (int): Frame index, starting from 0.
    k_mat (numpy.array): Camera intrinsic matrix, dim = (3, 3).
  
  """

  def __init__(self, config, k_mat, init_egomo_vec=None):
    """Model initialization.
    
    Args:
      config (dict): Configuration of the whole inference pipeline, including
                     necessary frozen PB and parameters for Tensorflow models.
      k_mat (numpy.array): Camera intrinsic matrix, dim = (3, 3).
      init_egomo_vec (numpy.array): Initial pose of camera, dim = (6,).
    
    """
    self.det = detector(config['detector'])
    self.est = estimator(config['estimator'])
    self.t_cfg = config['tracker']
    self.t_list = []
    self.tid_new = 0
    self.image_seq = []
    self.egomo_trmat = None
    self.frame_idx = -1
    self.k_mat = k_mat
    if init_egomo_vec is None:
      self.init_egomo_vec = np.zeros(6)
    else:
      self.init_egomo_vec = np.array(init_egomo_vec)
     
  def run_frame(self, image):
    """Frame routine, including main pipeline, triplet buil-up, and trajectory
    pipeline.
    
    Args:
      image (numpy.array): Image array, dim = (h, w, c).
      
    Return:
      frame_idx (int): Frame index.
      disp (numpy.array): Disparity map, for visualization, dim = (h, w, c).
      egomo_trmat (numpy.array): Accumulated egomotion transformation matrix, 
                                 for visualization, dim = (4, 4).
      t_list (list of tracker): List of trackers for visualization.
    
    """
    self.frame_idx += 1
    # run main pipeline
    t0 = datetime.now()
    disp = self.main_pipeline(image)
    t1 = datetime.now()
    logging.info('main pipeline: {}'.format(get_tdiff(t0, t1)))
    
    # prepare image sequence of 3 for trajectory pipeline
    t0 = datetime.now()
    self.image_seq.append(image)
    if len(self.image_seq) > 3:
      del self.image_seq[0]
    t1 = datetime.now()
    logging.info('image stack: {}'.format(get_tdiff(t0, t1)))

    # run trajectory pipeline
    t0 = datetime.now()
    if len(self.image_seq) >= 3:
      self.egomo_trmat = self.traj_pipeline(prev_trmat=self.egomo_trmat)
    t1 = datetime.now()
    logging.info('traj pipeline: {}'.format(get_tdiff(t0, t1)))
    return self.frame_idx, disp, self.egomo_trmat, self.t_list
  
  def main_pipeline(self, image):
    """Main pipeline of tracking-by-detection.
    
    From one image, we can obtain a list of detected objects along with their
    bounding boxes, labels, and depth. Objects are tracked with the data
    association solver and a list of trackers.

    Args:
      image (numpy.array): Image array, dim = (h, w, c).
      
    Return:
      disp (numpy.array): Disparity map, for visualization, dim = (h, w, c).
    
    """
    # detection
    t0 = datetime.now()
    bbox_list, score_list, label_list = self.det.inference(image)
    t1 = datetime.now()
    logging.info('main pipeline (det): {}'.format(get_tdiff(t0, t1)))
  
    # estimation
    t0 = datetime.now()
    disp = self.est.inference(image)
    depth_list = self.est.calc_depth(bbox_list)
    t1 = datetime.now()
    logging.info('main pipeline (est): {}'.format(get_tdiff(t0, t1)))
    
    # tracker predict
    t0 = datetime.now()
    for t in self.t_list:
      t.predict()
    t1 = datetime.now()
    logging.info('main pipeline (trk_pred): {}'.format(get_tdiff(t0, t1)))
    
    # associate
    t0 = datetime.now()
    matched_pair, unmatched_bbox_list, _ = associate(bbox_list, label_list, self.t_list)
    t1 = datetime.now()
    logging.info('main pipeline (da_solver): {}'.format(get_tdiff(t0, t1)))
    
    t0 = datetime.now()
    # update trackers for matched_pair
    for m in matched_pair:
      t = self.t_list[m[1]]
      bbox = bbox_list[m[0]]
      depth = depth_list[m[0]]
      est_dict = {
          'label': label_list[m[0]],
          'score': score_list[m[0]]}
      t.update(self.frame_idx, bbox, depth, est_dict)
    
    # update in-track status of all trackers
    for t in self.t_list:
      t.update_status(self.frame_idx)
    
    # purge out dead trackers
    self.t_list = [t for t in self.t_list if t.get_status()]

    # create new trackers for unmatched_bbox_list
    for b_idx in unmatched_bbox_list:
      bbox = bbox_list[b_idx]
      depth = depth_list[b_idx]
      est_dict = {
          'label': label_list[b_idx],
          'score': score_list[b_idx]}
      self.t_list.append(tracker(self.t_cfg, self.tid_new, bbox, depth, est_dict))
      self.tid_new += 1

    t1 = datetime.now()
    logging.info('main pipeline (trk_upd): {}'.format(get_tdiff(t0, t1)))

    # disparity map for display
    return disp

  def traj_pipeline(self, prev_trmat=None):
    """Trajectory pipeline of tracking-by-detection.
    
    Given a previous egomotion transformation matrix and a triplet of images,
    we can obtain the egomotion for the new frame and accumulate it on previous
    pose to generate 3D coordinate transformation matrix. Then all objects are
    projected to 3D coordinate to generate absolute trajectories. Those 
    trajectories are stored in the dictionary of each tracker.

    Args:
      prev_trmat (numpy.array): Previously accumulated egomotion transformation
                                matrix, dim = (4, 4).
      
    Return:
      egomo_trmat (numpy.array): Updated egomotion transformation matrix, dim 
                                 = (4, 4).
    
    """
    # image_seq = [image(frame_idx-2), image(frame_idx-1), image(frame_idx)]
    # egomotion update
    egomo = self.est.get_egomotion(self.image_seq)

    # egomotion transformation
    assert self.frame_idx >= 2, 'invalid self.frame_idx'
    if prev_trmat is None:
      assert self.frame_idx == 2, 'invalid self.frame_idx'
      # initialization of ego transformation matrix
      init_trmat = egomo_vec2mat(self.init_egomo_vec)
      prev_trmat = np.matmul(init_trmat, egomo_vec2mat(egomo[0])) # frame 0 to 1
    egomo_trmat = np.matmul(prev_trmat, egomo_vec2mat(egomo[1]))

    # tracker list update
    for t in self.t_list:
      # skip lost trackers
      if t.get_status()==False:
        continue
      # bounding box & depth
      bbox, depth = t.get_bbox(), t.get_depth()
      # project to 3d camera coordinate
      p3d_cam = cam_proj(self.k_mat, bbox, depth)
      # transform to world coordinate
      p3d = coord_transform(egomo_trmat, p3d_cam)
      t.add_attr_to_est_dict('traj', p3d)
    
    return egomo_trmat
  
