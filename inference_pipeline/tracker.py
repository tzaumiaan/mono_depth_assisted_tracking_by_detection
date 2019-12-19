# -*- coding: utf-8 -*-
"""Tracker module.

This module contains a Kalman filter class as the generic filter, and a tracker
class for the object tracking in image plane, containing Kalman filter 
instances, in-track records, and a dictionary for attributes.

"""
from __future__ import division

# package dependency
import numpy as np
from scipy.linalg import inv
from absl import logging

class tracker():
  """Tracker class.
  
  This class works as an entity of a tracked object, containing the filters for
  the bounding box on image plan and its depth information, the tracker ID, the
  in-track status, and the dictionary for extra attributes which are not parts
  of filtering algorithm.
  
  Attributes:
    tid (int): A unique ID of a tracker.
    f_bbox (kalman_filter): Kalman filter for filtering the bounding box.
    f_depth (kalman_filter): Kalman filter for filtering the depth information.
    est_dict (dict): Dictionary for tracker attributes.
    last_update (int): Frame index of the last update.
    in_track (bool): In-track status, false for out-of-track.
  
  """

  def __init__(self, config, tid, bbox, depth, est_dict):
    """__init__ method.
    
    Args:
      config (dict): Configuration of tracker.
      tid (int): A unique ID of a tracker.
      bbox (list): Bounding box in a list as [xmin, ymin, xmax, ymax].
      depth (float): Estimated depth.
      est_dict (dict): Dictionary for tracker attributes.
    
    """
    assert bbox.shape[0]==4
    self.tid = tid
    self.f_bbox = kalman_filter(x_pos=np.array(bbox), rounding=True,
        state_cov=config['bbox_cov'], meas_cov=config['bbox_cov'])
    # note: the state_cov and meas_cov has to be adjusted 
    #       if the pixel2meter_scale changes.
    #       the best way is to calculate actual variance from the estimator outputs
    self.f_depth = kalman_filter(x_pos=np.array([depth]), rounding=False,
        state_cov=config['depth_cov'], meas_cov=config['depth_cov'])
    self.est_dict = est_dict
    self.last_update = 0
    self.in_track = False
  
  def predict(self):
    """Predict method. All filters perform predict method."""
    b_ = self.f_bbox.predict()
    d_ = self.f_depth.predict()

  def update(self, time_stamp, bbox, depth, est_dict):
    """Update method.
    
    All filters perform update method, with attributes and time stamp saved.
    If the tracker does not receive updates, this method will then not be 
    called, leaving all these attributes not updated.
    
    Args:
      time_stamp (int): Frame index of this update.
      bbox (list): Bounding box in a list as [xmin, ymin, xmax, ymax].
      depth (float): Estimated depth.
      est_dict (dict): Dictionary for tracker attributes.
    
    """
    self.f_bbox.update(bbox)
    self.f_depth.update(depth)
    self.est_dict = est_dict
    self.last_update = time_stamp
  
  def add_attr_to_est_dict(self, key, value):
    """Add an attribute to dictionary `est_dict`.

    Args:
      key (str): Key of attribute.
      value (obj): Value of attribute.

    """
    self.est_dict[key] = value

  def get_bbox(self):
    """ Get bounding box.

    Return:
      bbox (list): Bounding box in a list as [xmin, ymin, xmax, ymax].
    
    """
    return self.f_bbox.x[:4]
  
  def get_depth(self):
    """ Get depth information.

    Return:
      depth (float): Tracked depth.
    
    """
    return self.f_depth.x[0]
  
  def get_est_dict(self):
    """ Get attributes.

    Return:
      est_dict (dict): Dictionary for tracker attributes.
    
    """
    return self.est_dict

  def update_status(self, time_stamp):
    """Status update method.

    This method checks the status of this tracker. If this tracker has not been
    updated once within `loos_track_threshold` frames, it will be considered as
    lost with `in_track` labeled `false`. Once labeled lost, it could be 
    deleted (or not) depending on the garbage collection implemented outsite
    tracker class.

    Args:
      time_stamp (int): Frame index of this update.

    """
    loose_track_threshold = 5
    self.in_track = False
    if(time_stamp - self.last_update < loose_track_threshold):
      self.in_track = True
  
  def get_status(self):
    """ Get tracker status.

    Return:
      in_track (bool): In-track status, false for out-of-track.
    
    """
    return self.in_track


class kalman_filter():
  def __init__(self,
               x_pos=np.array([0]),
               state_model='const_velo',
               state_cov=10.0,
               proc_cov=1.0,
               meas_model='pos',
               meas_cov=1.0,
               rounding=False):
    logging.debug('Kalman filter initialization with init pos {}'.format(x_pos))
    # time step
    self.dt = 1
    
    # configuration
    self.rounding = rounding

    # state vector and state model
    x_dim = x_pos.shape[0]
    self.x_dim = x_dim
    self.x = np.concatenate((x_pos, np.zeros(x_dim))) # position and velocity
    self.P = state_cov * np.eye(x_dim*2)
    if state_model == 'const_velo':
      # x(k) = F x(k-1) + G a(k)
      # where a(k) is a random variable with 
      # Gaussian(0, proc_cov)
      # F = [ I I*dt]
      #     [ 0 I   ]
      self.F = np.block([
          [np.eye(x_dim), np.eye(x_dim)*self.dt],
          [np.zeros((x_dim,x_dim)), np.eye(x_dim)]])
      # G = (0.5*dt^2 dt)^T
      # Q = (G*G^T)*proc_cov = proc_cov * [I*(dt^4)/4 I*(dt^3)/4]
      #                                   [I*(dt^3)/2 I*(dt^2)  ]
      self.Q = proc_cov * np.block([
          [np.eye(x_dim)*(self.dt**4)/4, np.eye(x_dim)*(self.dt**3)/2],
          [np.eye(x_dim)*(self.dt**3)/2, np.eye(x_dim)*(self.dt**2)]])
    else:
      raise(ValueError, 'invalid state model')
    
    # measurement model
    if meas_model == 'pos':
      # H = [I 0]
      self.H = np.block([np.eye(x_dim), np.zeros((x_dim,x_dim))])
      self.R = meas_cov * np.eye(x_dim)
    else:
      raise(ValueError, 'invalid measurement model')
  
  def round_pos(self):
    self.x[:self.x_dim] = np.round(self.x[:self.x_dim]) 

  def predict(self):
    # prior prediction
    self.x = self.F.dot(self.x)
    self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
    if self.rounding:
      self.round_pos()
    return self.x

  def update(self, z):
    # innovation
    y = z - self.H.dot(self.x)
    S = self.R + self.H.dot(self.P).dot(self.H.T)
    # Kalman gain
    K = self.P.dot(self.H.T).dot(inv(S))
    # posterior update
    self.x += K.dot(y)
    if self.rounding:
      self.round_pos()
    self.P -= K.dot(self.H).dot(self.P)
    return self.x, K


# testing code
def test(_):
  import random
  import matplotlib.pyplot as plt
  
  t = tracker(tid=0, bbox=np.array([0,0,0,0]), depth=0, est_dict={'label':0})
  x_bx, x_by, x_d = [], [], []
  z_bx, z_by, z_d = [], [], []
  z_bbox = np.array([2,3,4,5])
  z_depth = np.array([3.3])
  for i in range(50):
    print('time =', i)
    t.predict()
    x_bx.append(0.5*(t.get_bbox()[0] + t.get_bbox()[2]))
    x_by.append(0.5*(t.get_bbox()[1] + t.get_bbox()[3]))
    x_d.append(t.get_depth())
    print('predection:\nbbox={}, depth={}'.format(t.get_bbox(), t.get_depth()))
    z_bx.append(0.5*(z_bbox[0] + z_bbox[2]))
    z_by.append(0.5*(z_bbox[1] + z_bbox[3]))
    z_d.append(z_depth.copy())
    print('obserzation:\nbbox={}, depth={}'.format(z_bbox, z_depth))
    t.update(time_stamp=i, bbox=z_bbox, depth=z_depth, est_dict={'label':0})
    print('posterior:\nbbox={}, depth={}'.format(t.get_bbox(), t.get_depth()))
    # object move
    z_bbox += np.array([20, 20, 40, 40]) # const speed
    z_bbox += np.random.randint(low=-10, high=10, size=4) # random walk
    z_depth += 0.4 # const speed
    z_depth += 0.5*np.random.randn() # random walk
  
  plt.plot(x_bx, x_by, marker='+')
  plt.plot(z_bx, z_by, marker='.')
  plt.legend(['x', 'z'], loc='upper left')
  plt.show()
  plt.plot(x_d, marker='+')
  plt.plot(z_d, marker='.')
  plt.legend(['x', 'z'], loc='upper left')
  plt.show()

if __name__ == "__main__":
  from absl import app, logging
  logging.set_verbosity(logging.DEBUG)
  app.run(test)

