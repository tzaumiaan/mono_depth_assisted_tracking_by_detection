"""Calculating the RMSE of egomotion."""
from __future__ import print_function
from __future__ import division

# package dependency
from absl import app, flags
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

# internal dependency
from geo_utils import egomo_vec2mat

def read_golden_txt(filename):
  pos_list = []
  rot_list = []
  with open(filename, 'r') as f:
    for line in f:
      str_vec = line.strip().split(',')
      str_vec = [s.strip() for s in str_vec]
      if str_vec[0] == 'None':
        pos, rot = None, None
      else:
        pos = [float(str_vec[i]) for i in range(3)]
        rot = [float(str_vec[i]) for i in range(3,6)]
      pos_list.append(pos)
      rot_list.append(rot)
  print('golden txt = {}, length = {}'.format(filename, len(pos_list)))
  return pos_list, rot_list

def read_result_csv(filename):
  frame_cnt = 2 # first valid frame with ego pose
  pos_list = [None] * frame_cnt
  rot_list = [None] * frame_cnt
  with open(filename, 'r') as f:
    for line in f:
      str_vec = line.strip().split(',')
      str_vec = [s.strip() for s in str_vec]
      if str_vec[1] == '-1':
        assert(int(str_vec[0]) == frame_cnt)
        frame_cnt += 1
        pos = [float(str_vec[i]) for i in range(2,5)]
        rot = [float(str_vec[i]) for i in range(5,8)]
        pos_list.append(pos)
        rot_list.append(rot)
  print('result csv = {}, length = {}'.format(filename, len(pos_list)))
  return pos_list, rot_list

def calc_diff(vec_list):
  assert len(vec_list) > 1
  diff_list = []
  for i in range(len(vec_list)-1):
    if vec_list[i] is None or vec_list[i+1] is None:
      diff = None
    else:
      diff = list(np.array(vec_list[i+1]) - np.array(vec_list[i]))
    diff_list.append(diff)
  return diff_list

def calc_trmat(pos_list, rot_list):
  assert len(pos_list) == len(rot_list)
  assert len(pos_list) > 1
  pos_diff_list, rot_diff_list = [], []
  for i in range(len(pos_list)-1):
    if pos_list[i] is None or pos_list[i+1] is None:
      pos_diff, rot_diff = None, None
    else:
      pose_old = egomo_vec2mat(np.array(list(pos_list[i] + rot_list[i])))
      pose_new = egomo_vec2mat(np.array(list(pos_list[i+1] + rot_list[i+1])))
      trmat = np.matmul(np.linalg.inv(pose_old), pose_new)
      pos_diff = np.linalg.norm(trmat[0:3,3])
      rot_diff = np.arccos(0.5*(np.trace(trmat[0:3,0:3])-1))
      rot_diff = 180*(rot_diff/np.pi)
      # huge jump regarded as invalid
      if np.abs(rot_diff) > 60:
        pos_diff, rot_diff = None, None
    pos_diff_list.append(pos_diff)
    rot_diff_list.append(rot_diff)
  return pos_diff_list, rot_diff_list

def vec_diff(a, b):
  diff = np.array(b) - np.array(a)
  e = np.sqrt(np.sum(diff**2))
  return e

def scalar_diff(a, b):
  return b - a

def calc_rmse(golden, result, diff_fn):
  n_list = min(len(golden),len(result))
  error = np.zeros(n_list)
  for i in range(n_list):
    if golden[i] is None or result[i] is None:
      error[i] = np.nan
      continue
    error[i] = diff_fn(golden[i], result[i])
  rmse = np.sqrt(np.nanmean(error**2))
  return rmse, error

def rel_pos_to_vel(rel_pos_list):
  vscale = 30 * 3.6 # 30 fps, 1 m/s = 3.6km/h
  vel_list = []
  for x in rel_pos_list: 
    if x is None:
      vel_list.append(None)
    else:
      vel_list.append(vscale*x)
  return vel_list

def main(_):
  golden_pos, golden_rot = read_golden_txt(flags.FLAGS.golden)
  result_pos, result_rot = read_result_csv(flags.FLAGS.result)
  # absolute translation error
  abs_pos_rmse, abs_pos_err= calc_rmse(golden_pos, result_pos, vec_diff)
  print('Abs Trans RMSE = {} (meter)'.format(abs_pos_rmse))
  # relative pose error
  golden_rel_pos, golden_rel_rot = calc_trmat(golden_pos, golden_rot)
  result_rel_pos, result_rel_rot = calc_trmat(result_pos, result_rot)
  rel_pos_rmse, rel_pos_err = calc_rmse(golden_rel_pos, result_rel_pos, scalar_diff)
  rel_rot_rmse, rel_rot_err = calc_rmse(golden_rel_rot, result_rel_rot, scalar_diff)
  print('Rel Trans RMSE = {} (meter)'.format(rel_pos_rmse))
  print('Rel Rot RMSE = {} (degree)'.format(rel_rot_rmse))
  # velocity
  vscale = 30 * 3.6 # 30 fps, 1 m/s = 3.6km/h
  golden_vel = rel_pos_to_vel(golden_rel_pos)
  result_vel = rel_pos_to_vel(result_rel_pos)
  # visualization
  fig, axs= plt.subplots(4, 1, sharex=True, figsize=(6,10), dpi=100)
  axs[0].scatter(range(len(abs_pos_err)), abs_pos_err, s=0.5, c='black')
  axs[0].set_ylim((0,120))
  axs[0].set_ylabel('meter')
  axs[0].set_title('Absolute Translation Error')
  axs[0].grid(linestyle=':')
  axs[1].scatter(range(len(result_vel)), result_vel, s=0.5, c='black')
  axs[1].scatter(range(len(golden_vel)), golden_vel, s=0.5, c='gray')
  axs[1].set_ylim((0,160))
  axs[1].set_ylabel('km/h')
  axs[1].set_title('Velocity, Est: black, GT: gray')
  axs[1].grid(linestyle=':')
  axs[2].scatter(range(len(rel_pos_err)), rel_pos_err, s=0.5, c='black')
  axs[2].set_ylim((-1,1))
  axs[2].set_ylabel('meter')
  axs[2].set_title('Relative Pose Error: Translation Part')
  axs[2].grid(linestyle=':')
  axs[3].scatter(range(len(rel_rot_err)), rel_rot_err, s=0.5, c='black')
  axs[3].set_ylim((-0.6,0.6))
  axs[3].set_ylabel('degree')
  axs[3].set_title('Relative Pose Error: Rotation Part')
  axs[3].set_xlabel('frame')
  axs[3].grid(linestyle=':')
  fig.tight_layout()
  plt.show()

if __name__ == "__main__":
  flags.DEFINE_string('golden', None, 'Path of golden odometry txt file.')
  flags.DEFINE_string('result', None, 'Path of trajectory summary csv file.')
  flags.mark_flag_as_required('golden')
  flags.mark_flag_as_required('result')
  app.run(main)
