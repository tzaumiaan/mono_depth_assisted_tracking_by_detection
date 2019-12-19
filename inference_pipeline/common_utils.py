"""Common utilities module.

Here collects several utilities for common purposes, including reading images,
reading configuration, reading Tensorflow models, as well as summarization of
tracks.

"""

# package dependency
import cv2
import json
import os
import numpy as np
from absl import logging
from datetime import datetime
import tensorflow as tf
from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# internal dependency
from geo_utils import trmat2vec

CONFIG_FILE = 'config.json'

def read_config(subset=None):
  """Reading configuration JSON file as dictionary output.

  Args:
    subset (str, optional): Subset of configuration. If None is given, return
                            the whole configuration.
  
  Return:
    config (dict): Dictionary format of configuration.
  
  """
  assert os.path.exists(CONFIG_FILE)
  config = json.load(open(CONFIG_FILE))
  if subset is None:
    return config
  elif subset in config:
    return config[subset]
  else:
    raise ValueError('invalid subset {}'.format(subset))

def read_image(file_path):
  """Reading image file to numpy array.

  Args:
    file_path (str): File path.
  
  Return:
    image (numpy.array): RGB image, dim = (h, w, c).
  
  """
  assert os.path.exists(file_path), '{}: file not found'.format(file_path)
  return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def read_json(file_path):
  """Reading json data as dict.
  
  Args:
    file_path (str): File path.
  
  Return:
    json_dict (dict): Dictionary format of json data.
  
  """
  with open(file_path, 'r') as f:
    json_dict = json.load(f)
  return json_dict

def get_neighbor_frame(file_path, frame_diff=0):
  """Getting the file path of a neighbor frame.

  Args:
    file_path (str): File path.
    frame_diff (int, optional): Frame difference, 1 means next frame and -1
                                means previous frame, default = 0.
  
  Return:
    target_path (str): File path of target frame.
  
  """
  assert os.path.exists(file_path), '{}: file not found'.format(file_path)
  img_path_list = file_path.split('/')
  img_file = img_path_list[-1]
  img_folder = os.path.join(*img_path_list[:-1])
  img_ = img_file.split('.')
  frame_count = int(img_[0])
  frame_count_len = len(img_[0])
  target_frame = frame_count + frame_diff
  assert target_frame >= 0, '{}: target frame < 0'.format(target_frame)
  target_file = str(target_frame).zfill(frame_count_len)+'.'+img_[1]
  return os.path.join(img_folder, target_file)

def read_cam_intr(file_path):
  """Reading camera intrinsic.

  Args:
    file_path (str): File path.
  
  Return:
    k (numpy.array): Camera intrinsic matrix, dim = (3, 3).
  
  """
  assert os.path.exists(file_path), '{}: file not found'.format(file_path)
  f = open(file_path, 'r')
  k_str = f.readlines()[0].strip()
  k_str_list = k_str.split(',')
  k = np.array(k_str_list).astype(np.float)
  return k.reshape((3,3))

class tf_model(object):
  """Tensorflow model base class.
  
  This class works as the base class for Tensorflow inference. It provides the
  initialization flow for preparing the graph, as well as the unitily function
  of logging.
  
  Attributes:
    name (str): Name of this instance.
    graph (tf.Graph): Graph instance.
    sess (tf.Session): Session for inference.
  
  """
  def __init__(self, config):
    """__init__ method.
    
    Args:
      config (dict): Configuration containing necessary information for
                     serving, including frozen PB and necessary parameters.
    
    """
    self.name = config['name']
    self.graph = tf.Graph()
    tf_cfg = tf.ConfigProto(
        device_count={'GPU': 4}  # specify number of gpus used maximum
    )
    self.sess = tf.Session(graph=self.graph, config=tf_cfg)
    with self.graph.as_default():
      if config['using_pb']:
        # Load a frozen Tensorflow model into memory as graph
        t0 = datetime.now()
        graph_def_ = tf.GraphDef()
        pb_file = os.path.join(config['path'], config['pb_file'])
        with tf.gfile.GFile(pb_file, 'rb') as fid:
          graph_def_.ParseFromString(fid.read())
          tf.import_graph_def(graph_def_, name='')
        t1 = datetime.now()
        self._log_debug('Frozen graph imported from {}'.format(pb_file))
      else:
        # Load model from checkpoint
        t0 = datetime.now()
        init_op = tf.global_variables_initializer()
        ckpt_path = os.path.join(config['path'], config['ckpt_name'])
        meta_file = ckpt_path+'.meta'
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
        self.sess.run(init_op)
        saver.restore(self.sess, ckpt_path)
        t1 = datetime.now()
        self._log_debug('Meta graph imported from {}'.format(meta_file))
        self._log_debug('Weights restored from {}'.format(ckpt_path))
      self._log_info('Tensorflow model loaded')
      self._log_info('Elapsed time = {}sec'.format((t1-t0).total_seconds()))
  
  def _log_info(self, msg):
    """logging method with info level.
    
    Args:
      msg (str): Message to be printed.
    
    """
    logging.info('{}: {}'.format(self.name, msg))
  
  def _log_debug(self, msg):
    """logging method with debug level.
    
    Args:
      msg (str): Message to be printed.
    
    """
    logging.debug('{}: {}'.format(self.name, msg))


def summarize_frame(frame_idx, 
                    image, disp, traj_map, egomo_trmat, tracker_list, 
                    output_dir, vis_en, f_out):
  """Summarizing one frame as text report and 3 views: image, disparity, trajectory.
  
  Args:
    frame_idx (int): Frame index starting from 0.
    image (numpy.array): RGB image, dim = (h, w, c).
    disp (numpy.array): Disparity map of image, dim = (h, w, c).
    traj_map (numpy.array): Trajectory map, dim = (h_map, w_map, c).
    egomo_trmat (numpy.array): Egomotion transformation matrix, dim = (4, 4).
    tracker_list (list fo tracker): List of trackers.
    output_dir (str): Output directory for image files.
    vis_en (bool): Visualizaion results enabled.
    f_out (file or io.TextIOWrapper): File object for csv summary.
    
  Return:
    new_traj_map (numpy.array): Newl overlay of trjectory map, dim = (h_map, 
                                w_map, c).
  
  """
  from label_map import label_map
  from vis_utils import visualize_trk, visualize_traj
  
  # print out ego trajectory and rotation vector
  if egomo_trmat is not None:
    ego_tv, ego_rv = trmat2vec(egomo_trmat)
    info = 'Ego traj={}, eular rotvec={}'.format(ego_tv, ego_rv)
    logging.info('summary: {}'.format(info))
    if f_out is not None:
      common_field = '{: 10d}, {: 6d}, {: 8.4f}, {: 8.4f}, {: 8.4f}, '.format(
          frame_idx, -1,
          ego_tv[0], ego_tv[1], ego_tv[2])
      ego_field = '{:8.4f}, {:8.4f}, {:8.4f}, '.format(ego_rv[0], ego_rv[1], ego_rv[2])
      tracker_field = '{:10s}, {:8s}, {:5s}, {:5s}, {:5s}, {:5s}, {:8s}'.format('','','','','','','')
      f_out.write('{}{}{}\n'.format(common_field, ego_field, tracker_field))

  # print out in-track objects
  for t in tracker_list:
    common_field = '{: 10d}, {: 6d}, {:8s}, {:8s}, {:8s}, '.format(
        frame_idx, t.tid, '', '', '')
    ego_field = '{:8s}, {:8s}, {:8s}, '.format('','','')
    tracker_field = '{:10s}, {:8s}, {:5s}, {:5s}, {:5s}, {:5s}, {:8s}'.format('','','','','','','')
    if not t.get_status():
      info = 'Tracker{} inactive...'.format(t.tid)
    else:
      est_dict_ = t.get_est_dict()
      l_ = label_map[est_dict_['label']]
      s_ = est_dict_['score']
      d_ = t.get_depth()
      b_ = t.get_bbox()
      info = 'Tracker{}={}, score={:.3f}, depth={:.4f}, bbox={}'.format(t.tid, l_, s_, d_, b_)
      tracker_field = '{:10s}, {: 8.5f}, {: 5d}, {: 5d}, {: 5d}, {: 5d}, {: 8.5f}'.format(
          l_, s_, int(b_[0]), int(b_[1]), int(b_[2]), int(b_[3]), d_)
      if egomo_trmat is not None:
        t_ = est_dict_['traj']
        info += ', traj={}'.format(t_)
        common_field = '{: 10d}, {: 6d}, {: 8.4f}, {: 8.4f}, {: 8.4f}, '.format(
            frame_idx, t.tid, t_[0], t_[1], t_[2])
    logging.info('summary: {}'.format(info))
    if f_out is not None:
      f_out.write('{}{}{}\n'.format(common_field, ego_field, tracker_field))

  # visualization
  if vis_en:
    outfile = os.path.join(output_dir, 'main_frame_{}.png'.format(str(frame_idx).zfill(5)))
    visualize_trk(image, tracker_list, outfile)
    outfile = os.path.join(output_dir, 'disp_frame_{}.png'.format(str(frame_idx).zfill(5)))
    visualize_trk(disp, tracker_list, outfile) 
    outfile = os.path.join(output_dir, 'traj_frame_{}.png'.format(str(frame_idx).zfill(5)))
    traj_map = visualize_traj(egomo_trmat, tracker_list, traj_map, outfile)
  else:
    trag_map = None

  return traj_map

def png_to_video(out_dir, out_name, fps=30):
  """Collecting all PNG views from output directory to video clip.
  
  Args:
    out_dir (str): Output directory for image files.
    out_name (str): Output video file name.
  
  """
  image_folder = os.path.join(out_dir)
  video_name = os.path.join(out_dir, out_name+'.avi')
  prefix_list = ['main', 'disp', 'traj']
  # image list for each prefix
  images = {}
  for p_ in prefix_list:
    images_ = [i for i in os.listdir(image_folder) if i.startswith(p_) and i.endswith('png')]
    images[p_] = sorted(images_)
  # initialize video writer
  frame = cv2.imread(os.path.join(image_folder, images[prefix_list[0]][0]))
  frame_count = len(images[prefix_list[0]])
  h, w, _ = frame.shape # main
  h *= 2 # main+disp
  w += h # main+disp+traj
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  video = cv2.VideoWriter(video_name, fourcc=fourcc, fps=fps, frameSize=(w, h))
  # frame loop
  for i in range(frame_count):
    frame_p_list = [os.path.join(image_folder, images[p_][i]) for p_ in prefix_list]
    frames = [cv2.imread(f_) for f_ in frame_p_list]
    frames[2] = cv2.resize(frames[2], (h, h), interpolation=cv2.INTER_NEAREST)
    frame_cat = np.concatenate([frames[0], frames[1]], axis=0) # main+disp
    frame_cat = np.concatenate([frame_cat, frames[2]], axis=1) # main+disp+traj
    video.write(frame_cat)
  cv2.destroyAllWindows()
  video.release() 

def get_tdiff(t0, t1):
  """Turn two datetime format into a string of time difference."""
  return 'Elapsed time = {:.2f} ms'.format((t1-t0).total_seconds()*1000)

