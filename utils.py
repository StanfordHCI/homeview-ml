from glob import glob
import numpy as np
import config
import open3d
import math
import cv2
import os
import re

## file list sort function
def fn(file):
  file_split = re.split('[-.]', os.path.basename(file))
  if file_split[1] == 'json':
    return int(file_split[0])
  else:
    return int(file_split[0]) * 100 + int(file_split[1])


## 
def group_file_list(files):
  return [files[i:i+config.n_cameras] for i in range(0, len(files), config.n_cameras)]


## get file list for geometry, color and IoT data
def get_file_lists(dataset_name):
  
  ## if config.n_cameras exceeds 100, utils.py:fn has to be modified
  coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key = fn)
  n_frames = len(coord_files) // config.n_cameras
  ## ensures a minimum train-test split
  assert n_frames > config.n_train

  color_files = sorted(glob(dataset_name + '/raw/*.png'), key = fn)
  assert n_frames == len(color_files) // config.n_cameras

  sensor_files = sorted(glob(dataset_name + '/raw/*.json'), key = fn)
  ## some datasets contain an exceeding number of IoT data files
  # sensor_files = sensor_files[0:n_frames]
  assert n_frames == len(sensor_files)
  
  return group_file_list(coord_files), group_file_list(color_files), sensor_files, n_frames


## assemble open3d cloud, assuming colors have been scaled to [0,1]
def get_cloud(coords, colors):
  cloud = open3d.geometry.PointCloud()
  cloud.points = open3d.utility.Vector3dVector(coords)
  cloud.colors = open3d.utility.Vector3dVector(colors)
  return cloud


## horizontally split the entire space into chunks
def get_chunks(coords, size = 1):

  points = open3d.utility.Vector3dVector(coords)

  # bounding box of the entire space
  eps = 1e-2
  bmin, bmax = coords.min(0), coords.max(0)
  ymin, ymax = bmin[1] - eps, bmax[1] + eps

  xmin, xmax = math.floor(bmin[0] - eps), math.floor(bmax[0] + eps)
  zmin, zmax = math.floor(bmin[2] - eps), math.floor(bmax[2] + eps)

  chunks = []

  ## chunk as open3d bounding box
  for x in np.arange(xmin, xmax + 1, size):
    for z in np.arange(zmin, zmax + 1, size):
      chunk = open3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
        open3d.geometry.AxisAlignedBoundingBox(
          np.array([x, ymin, z]), np.array([x + size, ymax, z + size])
        )
      )
      ## ensures the chunk contains points
      idx = chunk.get_point_indices_within_bounding_box(points)
      if len(idx) > 0:
        chunks.append(chunk)

  return chunks


## stack points from multiple cameras
def get_coords_and_colors(coord_files, color_files):

  ## (n_points, 3) xyz
  coords = np.array([
    cv2.imread(coord_file, cv2.IMREAD_UNCHANGED)[:,:,:3]
    for coord_file in coord_files
  ]).reshape(-1, 3)

  ## (n_points, 3) rgb, [0, 255]
  colors = np.array([
    cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
    for color_file in color_files
  ]).reshape(-1, 3)

  return coords, colors