"""Geometric utilities module.

Here collects several utilities for geometric projection, including projection
between image plane and 3D world coordinate, conversion between vector 
representation and matrix one, and so on.

"""
from __future__ import division

# package dependency
import numpy as np
from scipy.spatial.transform import Rotation as R

def rot_vec2mat(rot_vec):
  """Conversion from rotation vector to rotation matrix.

  Args:
    rot_vec (numpy.array): Rotation vector as [rx, ry, rz], dim = (3,).
  
  Return:
    rot_mat (numpy.array): Rotation matrix, dim = (3, 3).
  
  """
  # rearrange rot_vec = [rx, ry, rz] to zyx order
  rv = np.array([rot_vec[2], rot_vec[1], rot_vec[0]])
  return R.from_euler('zyx', rv).as_dcm()

def egomo_vec2mat(egomo_vec):
  """Conversion from egomotion pose vector to transformation matrix.

  Args:
    egomo_vec (numpy.array): Egomoion pose vector as [tx, ty, tz, rx, ry, rz],
                             dim = (6,).
  
  Return:
    mat (numpy.array): Egomotion transformation matrix, dim = (4, 4).
  
  """
  # 4x4 transformation matrix
  mat = np.zeros((4,4)).astype(np.float)
  mat[:3,:3] = rot_vec2mat(egomo_vec[3:])
  mat[:3,3] = egomo_vec[:3]
  mat[3,3] = 1
  return mat

def trmat2vec(trmat):
  """Conversion from transformation matrix to translation and rotation vectors.

  Args:
    trmat (numpy.array): Transformation matrix, dim = (4, 4).
  
  Return:
    tv (numpy.array): Translation vector as [tx, ty, tz], dim = (3,).
    rv (numpy.array): Rotation vector as [rx, ry, rz], dim = (3,).
  
  """
  tv = np.array(trmat[:3,3])
  rv = R.from_dcm(trmat[:3,:3]).as_rotvec()
  return tv, rv

def cam_proj(k_mat, bbox, depth):
  """Camera projection from bounding box and depth to 3D point.

  Args:
    k_mat (numpy.array): Camera intrinsic matrix, dim = (3, 3).
    bbox (list): Bounding box in a list as [xmin, ymin, xmax, ymax].
    depth (float): Estimated depth.
  
  Return:
    p3d (numpy.array): 3D point at camera coordinate as [x, y, z], dim = (3,).
  
  """
  p2d = np.array([(bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5])
  p3d = np.zeros(3).astype(np.float)
  p3d[2] = depth
  p3d[0] = (p2d[0] - k_mat[0,2])/k_mat[0,0] * p3d[2];
  p3d[1] = (p2d[1] - k_mat[1,2])/k_mat[1,1] * p3d[2];
  return p3d

def coord_transform(trmat, p3d_orig):
  """Coordinate transform.

  Args:
    trmat (numpy.array): Transformation matrix, dim = (4, 4).
    p3d_orig (numpy.array): 3D point at original coordinate as [x, y, z], dim 
                            = (3,).
  
  Return:
    p3d_new (numpy.array): 3D point at new coordinate as [x, y, z], dim = (3,).
  
  """
  p4d_orig = np.append(p3d_orig, [1])
  p4d_new = trmat.dot(p4d_orig.T)
  return p4d_new[:3]

# testing
if __name__=='__main__':
  rot_vec = np.array([np.pi/2, np.pi/2, np.pi/2])
  rot_mat = rot_vec2mat(rot_vec)
  v, target_v = np.array([1,0,0]), np.array([0,0,1])
  diff = np.sum((target_v - rot_mat.dot(np.array(v.T)))**2)
  print(diff)
  v, target_v = np.array([0,1,0]), np.array([0,-1,0])
  diff = np.sum((target_v - rot_mat.dot(np.array(v.T)))**2)
  print(diff)

  egomo_vec = np.array([0.1, 0.2, 0.3, 0, 0, 0])
  egomo_mat = egomo_vec2mat(egomo_vec)
  print(egomo_mat)
  print(np.matmul(egomo_mat, egomo_mat))

  print(trmat2vec(egomo_mat))
