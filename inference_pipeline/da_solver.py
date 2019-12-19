# -*- coding: utf-8 -*-
"""Data association solver module.

Data association is the key to match a list of observed objects and the list of
currently tracked items. It can be reduced to linear assignment problem and 
solved by Hungarian algorithm (or Kuhn–Munkres algorithm). `scipy` package 
already implements this algorithm so we can directly use it.

"""

# package dependency
import numpy as np
from scipy.optimize import linear_sum_assignment

def associate(bbox_list, label_list, tracker_list, metric_thr=0.1):
  """Association solver method.
  
  Given a list of bounding boxes (with their labels) and a list of trackers,
  this method returns the indices of matched pairs and unmatched items.
  The metric includes IoU on bounding boxes and label consistency check.

  Args:
    bbox_list (list of list): List of bounding boxes (one bounding box is a 
                              list of as [xmin, ymin, xmax, ymax]).
    label_list (list of int): List of label from detector.
    tracker_list (list fo tracker): List of trackers.
    metric_thr (float, optional): Metric threshold for matching, default 0.1.
  
  Return:
    match_idx_pair (numpy.array): Matched indices pairs, dim = (n_pairs, 2).
    unmatched_bbox_idx (numpy.array): Unmatched bounding box indices, dim = 
                                      (n_ub,).
    unmatched_tracker_idx (numpy.array): Unmatched tracker indices, dim = 
                                         (n_ut,).
  
  """
  assert len(bbox_list) == len(label_list)

  # compute match matrix
  match_matrix = np.zeros((len(bbox_list),len(tracker_list))).astype(np.float32)
  for b_idx, bbox in enumerate(bbox_list):
    for t_idx, tracker in enumerate(tracker_list):
      if label_list[b_idx] == tracker.get_est_dict()['label']:
        match_matrix[b_idx,t_idx] = iou(bbox, tracker.get_bbox())
  # solve linear assignment
  # each entry is interpreted as the cost for that assignment
  # so IoU is taken with minus
  match_row, match_col = linear_sum_assignment(-match_matrix) 
  match_idx_pair_raw = list(zip(match_row, match_col))
  # kick out the match under threshold
  match_idx_pair = []
  for m in match_idx_pair_raw:
    if match_matrix[m[0],m[1]] > metric_thr:
      match_idx_pair.append(m)
  # keep this as np.array with fix dimension for convenience 
  if(len(match_idx_pair)==0):
    match_idx_pair = np.empty((0,2),dtype=int)
  else:
    match_idx_pair = np.array(match_idx_pair)

  # pick out the unmatched bbox
  unmatched_bbox_idx = []
  for b_idx, bbox in enumerate(bbox_list):
    if b_idx not in match_idx_pair[:,0]:
      unmatched_bbox_idx.append(b_idx)
  unmatched_bbox_idx = np.array(unmatched_bbox_idx)
  
  # pick out the unmatched tracker
  unmatched_tracker_idx = []
  for t_idx, tracker in enumerate(tracker_list):
    if t_idx not in match_idx_pair[:,1]:
      unmatched_tracker_idx.append(t_idx)
  unmatched_tracker_idx = np.array(unmatched_tracker_idx)

  return match_idx_pair, unmatched_bbox_idx, unmatched_tracker_idx


def iou(bbox_a, bbox_b):
  """IoU method.
  
  Compute the intersection over union of 2 bounding boxes.

  Args:
    bbox_a (list): Bounding box A in a list as [xmin, ymin, xmax, ymax].
    bbox_b (list): Bounding box B in a list as [xmin, ymin, xmax, ymax].
  
  Return:
    iou (float): Value of IoU, between [0, 1].
  
  """
  # utility functions
  def dx(bbox):
    return max(0, (bbox[2] - bbox[0] + 1))
  def dy(bbox):
    return max(0, (bbox[3] - bbox[1] + 1))

  # bbox area
  bbox_area_a = dx(bbox_a) * dy(bbox_a)
  bbox_area_b = dx(bbox_b) * dy(bbox_b)

  # intersection coordinates
  bbox_i = np.array([
      max(bbox_a[0], bbox_b[0]),
      max(bbox_a[1], bbox_b[1]),
      min(bbox_a[2], bbox_b[2]),
      min(bbox_a[3], bbox_b[3])])
  # intersection area
  intersec_area = dx(bbox_i) * dy(bbox_i)
  
  # union area
  union_area = bbox_area_a + bbox_area_b - intersec_area

  # intersection over union
  return float(intersec_area)/float(union_area)

# testing code
def test(_):
  from tracker import tracker

  # used for generating a random bounding box
  def gen_bbox():
    bbox = np.random.randint(0,99,size=4)
    bbox[0], bbox[2] = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
    bbox[1], bbox[3] = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
    return bbox
  
  bbox_list = []
  label_list = []
  tracker_list = []
  for i in range(flags.FLAGS.num):
    bbox = gen_bbox()
    label = i
    tracker_list.append(tracker(tid=i, bbox=bbox, depth=0, est_dict={'label': i}))
    bbox_list.append(bbox)
    label_list.append(label)
  
  logging.info('test perfect matching')
  m, ub, ut = associate(bbox_list, label_list, tracker_list)
  logging.info('m={}, ub={}, ut={}'.format(m, ub, ut))
  
  logging.info('test empty bbox list')
  m, ub, ut = associate((), (), tracker_list)
  logging.info('m={}, ub={}, ut={}'.format(m, ub, ut))
  
  logging.info('test empty tracker list')
  m, ub, ut = associate(bbox_list, label_list, ())
  logging.info('m={}, ub={}, ut={}'.format(m, ub, ut))
  
  bbox_list = []
  for i in range(flags.FLAGS.num):
    bbox_list.append(gen_bbox())
  
  logging.info('test random matching')
  m, ub, ut = associate(bbox_list, label_list, tracker_list)
  logging.info('m={}, ub={}, ut={}'.format(m, ub, ut))

if __name__ == "__main__":
  from absl import app, flags, logging
  flags.DEFINE_integer('num', 10, 'Number of runs.')
  logging.set_verbosity(logging.INFO)
  app.run(test)

