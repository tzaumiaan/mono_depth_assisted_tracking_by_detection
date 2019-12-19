# -*- coding: utf-8 -*-
"""Estimator module.

This module is the wrapper of Tensorflow inference graph serving as the depth
and egomotion estimator. The estimator can take one image as input and generate
the disparity map, take a list of bounding boxes and generate their median 
depth, or take a triplet of images and generate the egomotion vector.

"""
from __future__ import division

# package dependency
import numpy as np
import cv2
from datetime import datetime

# internal dependency
from common_utils import tf_model, get_tdiff

class estimator(tf_model):
  """Estomator class.
  
  This class works as the wrapper of the depth and egomotion estimator. It is
  initialized to load the Tensorflow inference graph. It has the `inference`
  method to generate disparity map, `calc_depth` to generate the depth of a 
  given list of bounding boxes, and `get_egomotion` to generate the egomotion
  vector.
  
  Attributes:
    name (str, from tf_model): Name of this instance.
    graph (tf.Graph, from tf_model): Graph instance.
    sess (tf.Session, from tf_model): Session for inference.
    input_size (tuple of int): Width and height of neural network input.
    image_size (tuple of int): Width and height of the incoming image.
    disparity (numpy.array): Disparity map, dim = (h, w, c).
    egomotion (numpy.array): Egomotion vector [tx, ty, tz, rx, ry, rz], 
                             dim = (6,).
  
  """
  
  def __init__(self, config):
    """__init__ method.
    
    Args:
      config (dict): Configuration containing necessary information for
                     serving, including frozen PB and necessary parameters.
    
    """
    super(estimator, self).__init__(config)
    self.input_size = (config['img_width'], config['img_height'])
    self.pixel2meter_scale = config['pixel2meter_scale']

  def _scale(self, image):
    """Scale method to convert pixels of an image from [0, 255] to [0, 1]."""
    return np.array(image, dtype=np.float32) / 255.0

  def _resize(self, image, mode='inference'):
    """Resize method. Resize the image to input_size (inference mode) or 
    image_size (restore mode)."""
    if mode == 'inference':
      h, w, c = image.shape
      self.image_size = (w, h)
      target_size = self.input_size
    elif mode == 'restore':
      target_size = self.image_size
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

  def inference(self, image):
    """Inference method.
    
    This method corresponds to the disparity net of the struct2depth model.
    
    Args:
      image (numpy.array): Image array, dim = (h, w, c).
      
    Return:
      disparity (numpy.array): Disparity map, dim = (h, w, c).
    
    """
    h, w, c = image.shape
    # scale from 0-255 to 0-1
    image = self._scale(image)
    # image resize
    image_res = self._resize(image, mode='inference')
    image_batch = np.expand_dims(image_res, axis=0)
    # get operators from graph
    input_image = self.graph.get_tensor_by_name('depth_prediction/raw_input:0')
    output_disp = self.graph.get_tensor_by_name('depth_prediction/add_3:0')
    # run inference
    with self.graph.as_default():
      t0 = datetime.now()
      disp_batch = self.sess.run(output_disp, feed_dict={input_image: image_batch})
      t1 = datetime.now()
      self._log_info('*TF Disparity*: {}'.format(get_tdiff(t0, t1)))
      self.disparity = self._resize(np.squeeze(disp_batch), mode='restore')
      disp_scale = self.image_size[0]/self.input_size[0]
      self.disparity *= disp_scale
    return self.disparity
  
  def calc_depth(self, bbox_list):
    """Calculating depth from a given list of bounding boxes. Median measure is
    used to generate robust depth out of a bounding box.
    
    Args:
      bbox_list (numpy.array): List of bounding boxes, dim = (n, 4).
      
    Return:
      depth_list (numpy.array): List of depth, dim = (n,).
    
    """
    depth_list = []
    for bbox in bbox_list:
      [xmin, ymin, xmax, ymax] = bbox
      # take out the patch
      patch = self.disparity[ymin:ymax, xmin:xmax].astype(np.float64)
      # exclude outliers
      patch[patch<0] = np.nan
      # calculate the median of this patch
      median_disp = np.nanmedian(patch)
      # disparity to depth formula
      depth = 1 / median_disp
      depth *= self.pixel2meter_scale
      depth_list.append(depth)
    depth_list = np.array(depth_list)
    return depth_list
  
  def get_egomotion(self, image_list):
    """Obtaining egomotion vector from a triplet of images.
    
    Args:
      image_list (numpy.array): List of images, dim = (3, h, w, c).
      
    Return:
      egomotion (numpy.array): Egomotion vector [tx, ty, tz, rx, ry, rz],
                               dim = (6,).
    
    """
    image_list_res = [self._resize(i, mode='inference') for i in image_list]
    image_stack = np.concatenate(image_list_res, axis=2)
    image_stack = self._scale(image_stack) # scale from 0-255 to 0-1
    image_stack_batch = np.expand_dims(image_stack, axis=0)
    # get operators from graph
    input_image_stack = self.graph.get_tensor_by_name('raw_input:0')
    output_egomotion = self.graph.get_tensor_by_name('egomotion_prediction/pose_exp_net/pose/concat:0')
    # run inference
    with self.graph.as_default():
      t0 = datetime.now()
      egomotion_batch = self.sess.run(output_egomotion, feed_dict={input_image_stack: image_stack_batch})
      t1 = datetime.now()
      self._log_info('*TF Egomotion*: {}'.format(get_tdiff(t0, t1)))
      self.egomotion = np.squeeze(egomotion_batch)
      self.egomotion[:,:3] *= self.pixel2meter_scale
    return self.egomotion
    
# testing code
def test(_):
  from common_utils import read_config, read_image, get_neighbor_frame
  img = read_image(flags.FLAGS.image)
  e = estimator(read_config('estimator'))
  # disparity
  disp = e.inference(img)
  # egomotion
  prev_img = read_image(get_neighbor_frame(flags.FLAGS.image, frame_diff=-1))
  next_img = read_image(get_neighbor_frame(flags.FLAGS.image, frame_diff=1))
  img_list = [prev_img, img, next_img]
  egomo = e.get_egomotion(img_list)
  logging.info('egomotion prev-cur: {}'.format(egomo[0]))
  logging.info('egomotion cur-next: {}'.format(egomo[1]))
  # depth and visualization
  if flags.FLAGS.w_det:
    from detector import detector
    from vis_utils import visualize_det
    d = detector(read_config('detector'))
    bbox_list, score_list, label_list = d.inference(img)
    depth_list = e.calc_depth(bbox_list)
    visualize_det(disp, bbox_list, score_list, label_list, depth_list)
  else:
    from vis_utils import visualize_img
    visualize_img(disp)

if __name__=='__main__':
  import os
  os.environ['KMP_DUPLICATE_LIB_OK']='True' # to work around OpenMP multiple loading assertion
  
  from absl import app, flags, logging
  flags.DEFINE_string('image', None, 'Input image file.')
  flags.DEFINE_boolean('w_det', True, 'Test with object detector.')
  flags.mark_flag_as_required('image')
  logging.set_verbosity(logging.INFO)
  app.run(test)

