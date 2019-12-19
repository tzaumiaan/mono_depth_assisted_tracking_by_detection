# -*- coding: utf-8 -*-
"""Main entry for demo.

Usage:
  python main.py \
    [--input_dir ../data/kitti/20110926/image/0056] \
    [--output_dir output] \
    [--n_frames 294]

"""

# package dependency
from absl import app, flags, logging
import os, shutil
from datetime import datetime

# internal dependency
from inference_core import inference_wrapper
from common_utils import read_config, read_image, read_cam_intr, read_json
from common_utils import summarize_frame, png_to_video
from vis_utils import init_traj_map

def get_image_list(config): 
  """Getting image list from configuration. If parameters in configuration are
  not overwritten by flags then it follows the configuration file. Otherwise
  it follows the overwritten values.

  Args:
    config (dict): Configuration containing input path.
  
  Return:
    image_dir (str): Path of image directory.
    image_list (list of str): List of image file names.

  """
  if flags.FLAGS.input_dir is not None:
    config['input']['path'] = os.path.dirname(flags.FLAGS.input_dir)
    config['input']['clip'] = os.path.basename(flags.FLAGS.input_dir)
  image_dir = os.path.join(config['input']['path'], config['input']['clip'])
  assert os.path.exists(image_dir), 'path not found: {}'.format(image_dir)
  logging.info('main: reading images from {}'.format(image_dir))
  image_list = [i for i in os.listdir(image_dir) if i.endswith('png')]
  image_list = sorted(image_list)
  if flags.FLAGS.n_frames is not None:
    config['input']['max_frames'] = flags.FLAGS.n_frames
  if config['input']['max_frames'] > 0:
    image_list = image_list[:config['input']['max_frames']]
  logging.info('main: total {} frames'.format(len(image_list)))
  return image_dir, image_list

def get_cam_intr(image_dir): 
  """Getting camera intrinsic matrix from image directory.

  Args:
    image_dir (str): Path of image directory.
  
  Return:
    k_mat (numpy.array): Camera intrinsic matrix, dim = (3, 3).

  """
  import numpy as np
  # default path of calibration file
  dataset_dir = os.path.dirname(os.path.dirname(image_dir))
  calib_file = os.path.join(dataset_dir, 'calibration', 'calib.txt')
  logging.info('main: reading calibration file {}'.format(calib_file))
  return read_cam_intr(calib_file)

def get_racetrack(image_dir):
  """Getting racetrack data.

  Args:
    image_dir (str): Path of image directory.
  
  Return:
    racetrack (dict): Dictionary format of racetrack data.

  """
  import numpy as np
  # default path of calibration file
  dataset_dir = os.path.dirname(os.path.dirname(image_dir))
  racetrack_file = os.path.join(dataset_dir, 'racetrack.json')
  if not os.path.exists(racetrack_file):
    return None
  logging.info('main: reading racetrack file {}'.format(racetrack_file))
  racetrack = read_json(racetrack_file)
  return racetrack

def get_init_egomo_vec(image_dir, clip_name): 
  """Getting initial egomotion vector.

  Args:
    image_dir (str): Path of image directory.
    clip_name (str): Name of clip.
  
  Return:
    init_egomo_vec (numpy.array): Initial egomotion vector, dim = (6,).

  """
  import numpy as np
  # default path of calibration file
  dataset_dir = os.path.dirname(os.path.dirname(image_dir))
  egomo_file = os.path.join(dataset_dir, 'init_egomo_vec.json')
  if not os.path.exists(egomo_file):
    return np.zeros(6)
  logging.info('main: reading egomotion file {}'.format(egomo_file))
  egomo_dict = read_json(egomo_file)
  if clip_name not in egomo_dict:
    return np.zeros(6)
  init_egomo_vec = np.array(egomo_dict[clip_name])
  return init_egomo_vec

def create_output_dir(config):
  """Creating output directory from configuration. If parameters in configuration are
  not overwritten by flags then it follows the configuration file. Otherwise
  it follows the overwritten values.

  Args:
    config (dict): Configuration containing output path.
  
  Return:
    output_dir (str): Path of output directory.

  """
  if flags.FLAGS.output_dir is not None:
    output_dir = flags.FLAGS.output_dir
  else: 
    output_dir = config['output']['path']
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.mkdir(output_dir)
  logging.info('main: write to {}'.format(output_dir))
  return output_dir

def main(_):
  """Main entry for demo."""
  
  # configuration, input and output
  config = read_config()
  image_dir, image_list = get_image_list(config)
  k_mat = get_cam_intr(image_dir)
  racetrack = get_racetrack(image_dir)
  init_egomo_vec = get_init_egomo_vec(image_dir, config['input']['clip'])
  output_dir = create_output_dir(config)
  vis_en = config['output']['vis_en']
  csv_en = config['output']['csv_en']
  summary_file = os.path.join(output_dir, 'summary.csv')

  # model initialization
  inf_core = inference_wrapper(config, k_mat, init_egomo_vec)
  traj_map = init_traj_map(racetrack)
  if csv_en:
    f_out = open(summary_file, 'w')
    common_field = '{:10s}, {:6s}, {:8s}, {:8s}, {:8s}, '.format('frame_idx','tid','traj_x','traj_y','traj_z')
    ego_field = '{:8s}, {:8s}, {:8s}, '.format('rx','ry','rz')
    tracker_field = '{:10s}, {:8s}, {:5s}, {:5s}, {:5s}, {:5s}, {:8s} ,'.format('label', 'score', 'xmin','ymin','xmax','ymax','depth')
    f_out.write('{}{}{}\n'.format(common_field, ego_field, tracker_field))
  else:
    f_out = None
  logging.info('main: initialization done')
  
  # frame loop
  for frame_idx, image_file in enumerate(image_list):
    image = read_image(os.path.join(image_dir, image_file))
    t0 = datetime.now()
    frame_idx, disp, egomo_trmat, t_list = inf_core.run_frame(image)
    exec_time = (datetime.now() - t0).total_seconds()*1000
    logging.info('main: frame {} main part finished with exec time {:.2f} ms'.format(frame_idx, exec_time))
    traj_map = summarize_frame(frame_idx, image, disp, traj_map, egomo_trmat, t_list, output_dir, vis_en, f_out)
  if csv_en:
    f_out.close()
  logging.info('main: all images done')
   
  if vis_en:
    png_to_video(output_dir, 'main', fps=config['output']['vis_fps'])
    logging.info('main: video converted')


if __name__ == "__main__":
  import os
  os.environ['KMP_DUPLICATE_LIB_OK']='True' # to work around OpenMP multiple loading assertion
  
  flags.DEFINE_string('input_dir', None, 'Path for input image files.')
  flags.DEFINE_string('output_dir', None, 'Path for output visualization results.')
  flags.DEFINE_integer('n_frames', None, 'Max number of frames.')
  logging.set_verbosity(logging.INFO)
  app.run(main)


