"""Visualization utilities module.

Here collects several utilities for visualization, including drawing overlay 
of tracks on an image or a disparity map, drawing trajectories, as well as 
smaller building blocks.

"""
from __future__ import division

# package dependency
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import cv2

# internal dependency
from label_map import label_map

# constants
DPI = 80
LINESIZE = 4
TEXTSIZE = 16

RACETRACK_X_OFFSET = 50
RACETRACK_Z_OFFSET = 0

def _traj_to_pos(traj, h, w, scale=float(1), offset=(0,0)):
  """Converting 2D trjectory from world coordinate to canvas coordinate.

  Args:
    traj (numpy.array): Trajectory point as [x, z].
    h (int): Canvas height.
    w (int): Canvas width.
    scale (float, optional): Coordinate scaling factor, default = 1.
    offset (tuple, optional): Coordinate offset as (w, h), default = (0, 0).
  
  Return:
    rgb_color (tuple of float): RGB color as (R, G, B).

  """
  return (int(traj[0]*scale + w/2 + offset[0]), int(-traj[1]*scale + h/2 + offset[1]))

def _colorize(tracker_id):
  """Giving a unique color to a tracker based on tracker ID.

  Args:
    tracker_id (int): Tracker ID.
  
  Return:
    rgb_color (tuple of float): RGB color as (R, G, B).

  """
  if tracker_id=='ego':
    rgb_color = (0,0,0)
  else:
    hue = int(tracker_id) % 256 # mod 256 to fit 0-255
    hue = int('{:08b}'.format(hue)[::-1], 2) # scramble
    hue = float(hue)/256 # normalize
    sat = 0.8
    val = 1.0
    rgb_color = hsv_to_rgb((hue, sat, val))
  return rgb_color

def visualize_img(image, outfile=None):
  """Visualizing an image.

  Args:
    image (numpy.array): Image array, dim = (h, w, c).
    outfile (str, optional): Output file name, default is None, meaning
                             displaying in a pop-out window.

  """
  h, w = image.shape[0], image.shape[1]
  figsize = w / float(DPI), h / float(DPI)
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(image, cmap='plasma')
  if outfile is not None:
    fig.savefig(outfile, dpi=DPI) # save out
    print("VisUtils: {} saved".format(outfile))
  else:
    plt.show()
  # clear all the stuff before next frame
  fig.clf()
  plt.close()

def visualize_det(image, bbox_list, score_list=None, label_list=None, depth_list=None, outfile=None):
  """Visualizing detector result.

  Args:
    image (numpy.array): Image array, dim = (h, w, c).
    bbox_list (numpy.array): List of bounding boxes, dim = (n, 4).
    score_list (numpy.array, optional): List of confidence scores, dim = (n,).
                                        Default is None, meaning obmitted.
    label_list (numpy.array, optional): List of labels, dim = (n,). Default is
                                        None, meaning obmitted.
    depth_list (numpy.array, optional): List of depth, dim = (n,). Default is
                                        None, meaning obmitted.
    outfile (str, optional): Output file name, default is None, meaning
                             displaying in a pop-out window.

  """
  h, w = image.shape[0], image.shape[1]
  figsize = w / float(DPI), h / float(DPI)
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(image, cmap='plasma')
  for i, (bbox) in enumerate(bbox_list):
    color = _colorize(i)
    pos = (bbox[0], bbox[1])
    dx, dy = bbox[2]-bbox[0], bbox[3]-bbox[1]
    r = patches.Rectangle(pos,dx,dy,linewidth=LINESIZE,edgecolor=color,facecolor='none')
    ax.add_patch(r)
    text = 'bbox{}'.format(i)
    if label_list is not None:
      text += '\n{}'.format(label_map[label_list[i]])
    if score_list is not None:
      text += '({:0.3f})'.format(score_list[i]) 
    if depth_list is not None:
      text += '\ndepth={:0.4f}'.format(depth_list[i]) 
    ax.text(bbox[0],bbox[1], text, size=TEXTSIZE, color=color)
  if outfile is not None:
    fig.savefig(outfile, dpi=DPI) # save out
    print("VisUtils: {} saved".format(outfile))
  else:
    plt.show()
  # clear all the stuff before next frame
  fig.clf()
  plt.close()

def visualize_trk(image, tracker_list, outfile=None):
  """Visualizing tracker result.

  Args:
    image (numpy.array): Image array, dim = (h, w, c).
    tracker_list (list of tracker): List of trackers for visualization.
    outfile (str, optional): Output file name, default is None, meaning
                             displaying in a pop-out window.

  """
  h, w = image.shape[0], image.shape[1]
  figsize = w / float(DPI), h / float(DPI)
  fig = plt.figure(figsize=figsize)
  ax = fig.add_axes([0, 0, 1, 1])
  ax.axis('off')
  ax.imshow(image, cmap='plasma')
  for t in tracker_list:
    # skip lost trackers
    if t.get_status()==False:
      continue
    color = _colorize(t.tid)
    # bounding boxes
    bbox = t.get_bbox()
    pos = (bbox[0], bbox[1])
    dx, dy = bbox[2]-bbox[0], bbox[3]-bbox[1]
    r = patches.Rectangle(pos,dx,dy,linewidth=LINESIZE,edgecolor=color,facecolor='none')
    ax.add_patch(r)
    # estimation
    est_dict = t.get_est_dict()
    depth = t.get_depth()
    text = 'tracker{}\n{}({:.3f})\ndepth={:.4f}'.format(t.tid,
        label_map[est_dict['label']],
        est_dict['score'],
        depth)
    ax.text(bbox[0],bbox[1], text, size=TEXTSIZE, color=color)
  if outfile is not None:
    fig.savefig(outfile, dpi=DPI) # save out
    print("VisUtils: {} saved".format(outfile))
  else:
    plt.show()
  # clear all the stuff before next frame
  fig.clf()
  plt.close()

def init_traj_map(racetrack):
  """Initializing trajectory map.
  
  Args:
    racetrack (dict): Dictionary format of racetrack data.
  
  Return:
    traj_map (numpy.array): Trajectory map, dim = (h_map, w_map, c).
  
  """
  if racetrack is None:
      return None 

  h, w = 1000, 1000
  def draw_circuit(img, pts, color, ref=None):
    origin = np.array([int(img.shape[0] / 2), int(img.shape[1] / 2)])
    if ref is not None:
      origin += np.array(ref)
    for i in range(1,len(pts)):
      pt_start = (int(pts[i-1][0])+origin[0], int(-pts[i-1][1])+origin[1])
      pt_end = (int(pts[i][0])+origin[0], int(-pts[i][1])+origin[1])
      cv2.line(img, pt_start, pt_end, color)
  
  traj_map = np.ones((h, w, 3), np.uint8) * 255
  offset = (RACETRACK_X_OFFSET, RACETRACK_Z_OFFSET)
  draw_circuit(traj_map, racetrack[u'Inside'], (64, 64, 64), offset)
  draw_circuit(traj_map, racetrack[u'Outside'], (64, 64, 64), offset)
  draw_circuit(traj_map, racetrack[u'InsidePitlane'], (32, 32, 32), offset)
  draw_circuit(traj_map, racetrack[u'OutsidePitlane'], (32, 32, 32), offset)
  return traj_map

def visualize_traj(egomo_trmat, tracker_list, traj_map=None, outfile=None):
  """Visualizing trajectories.

  Args:
    egomo_trmat (numpy.array): Accumulated egomotion transformation matrix, 
                               for visualization, dim = (4, 4).
    tracker_list (list of tracker): List of trackers for visualization.
    traj_map (numpy.array, optional): Previous trajectory map, dim = (h_map,
                                      w_map, c). Default in None, meaning 
                                      drawing as a new map.
    outfile (str, optional): Output file name, default is None, meaning
                             displaying in a pop-out window.

  """
  # note: h/scale and w/scale should fit the range of trajectory
  h, w = 1000, 1000
  offset = (RACETRACK_X_OFFSET, RACETRACK_Z_OFFSET)
  if traj_map is None:
    # initialize traj_map
    traj_map = np.ones((h,w,3), np.uint8) * 255
  assert traj_map.shape == (h, w, 3)
  # draw ego trajectory
  traj_map_w_marker = traj_map.copy()
  if egomo_trmat is not None:
    ego_traj = np.array([egomo_trmat[0,3], egomo_trmat[2,3]])
    pos_ = _traj_to_pos(ego_traj, h, w, offset=offset)
    color_ = tuple([int(c_*255) for c_ in _colorize('ego')])
    cv2.circle(traj_map, pos_, 2, color_, -1)
    cv2.circle(traj_map_w_marker, pos_, 2, color_, -1)
    cv2.putText(traj_map_w_marker, 'ego', pos_, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_)
  # draw object trajectory
  for t in tracker_list:
    if t.get_status()==False:
      continue
    if 'traj' in t.est_dict:
      obj_traj = np.array([t.est_dict['traj'][0], t.est_dict['traj'][2]])
      pos_ = _traj_to_pos(obj_traj, h, w, offset=offset)
      color_ = tuple([int(c_*255) for c_ in _colorize(t.tid)])
      cv2.circle(traj_map, pos_, 2, color_, -1)
      cv2.circle(traj_map_w_marker, pos_, 2, color_, -1)
      l_ = '{}:{}'.format(t.tid, label_map[t.est_dict['label']])
      cv2.putText(traj_map_w_marker, l_, pos_, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_)
  if outfile is not None:
    img_ = cv2.cvtColor(traj_map_w_marker, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outfile, img_)
    print("VisUtils: {} saved".format(outfile))
  return traj_map

