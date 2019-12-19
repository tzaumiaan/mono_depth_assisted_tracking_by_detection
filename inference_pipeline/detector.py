# -*- coding: utf-8 -*-
"""Detector module.

This module is the wrapper of Tensorflow inference graph serving as the 
objection detector. The detector takes one image as input and generate the
bounding boxes, confidence scores, and labels of detected objects.

"""

# package dependency
import numpy as np
from datetime import datetime

# internal dependency
from common_utils import tf_model, get_tdiff

class detector(tf_model):
  """Detector class.
  
  This class works as the wrapper of object detector. It is initialized to 
  load the Tensorflow inference graph, and has the `inference` method to serve
  its functionality. Some of the infrastructures are inherited from `tf_model`
  class.
  
  Attributes:
    name (str, from tf_model): Name of this instance.
    graph (tf.Graph, from tf_model): Graph instance.
    sess (tf.Session, from tf_model): Session for inference.
    score_threshold (float): Threshold for an detected object to be reported.
  
  """
  
  def __init__(self, config):
    """__init__ method.
    
    Args:
      config (dict): Configuration containing necessary information for
                     serving, including frozen PB and necessary parameters.
    
    """
    super(detector, self).__init__(config)
    self.score_threshold = config['score_threshold']

  def inference(self, image, score_threshold=None):
    """Inference method.
    
    This method is the main function of this class. It takes in an image and 
    gives out detection results.
    
    Args:
      image (numpy.array): Image array, dim = (h, w, c).
      score_threshold (float, optional): Threshold for an detected object to be
                                         reported. Default: None.
      
    Return:
      boxes (numpy.array): List of bounding boxes, dim = (n, 4).
      scores (numpy.array): List of confidence scores, dim = (n,).
      classes (numpy.array): List of labels, dim = (n,).
    
    """
    h, w, c = image.shape
    image_batch = np.expand_dims(image, axis=0)
    # get operators from graph
    image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.graph.get_tensor_by_name('num_detections:0')
    # run inference
    with self.graph.as_default():
      t0 = datetime.now()
      (boxes, scores, classes, num) = self.sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_batch})
      t1 = datetime.now()
      num = int(num)
      self._log_info('*TF Detection*: {}'.format(get_tdiff(t0, t1)))
    # post processing ...
    # purge useless dimension 
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
    # take only valid results
    boxes, scores, classes = boxes[:num,:], scores[:num], classes[:num]
    # score threshold
    if score_threshold is None:
      score_threshold = self.score_threshold
    boxes = boxes[scores>score_threshold,:]
    classes = classes[scores>score_threshold]
    scores = scores[scores>score_threshold]
    num = scores.shape[0]
    self._log_info('{} objects found'.format(num))
    # x-y reorder
    boxes = boxes[:,np.array([1,0,3,2])]
    # transform from 0-1 to 0-w and 0-h
    boxes = np.multiply(boxes, np.array([w,h,w,h])).astype(np.int32)
    return boxes, scores, classes

# testing code
def test(_):
  from vis_utils import visualize_det
  from common_utils import read_config, read_image
  img = read_image(flags.FLAGS.image)
  d = detector(read_config('detector'))
  bbox_list, score_list, label_list = d.inference(img)
  visualize_det(img, bbox_list, score_list, label_list)

if __name__=='__main__':
  from absl import app, flags, logging
  flags.DEFINE_string('image', None, 'Input image file.')
  flags.mark_flag_as_required('image')
  logging.set_verbosity(logging.INFO)
  app.run(test)
