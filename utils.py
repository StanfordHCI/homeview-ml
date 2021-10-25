import open3d

import numpy as np
import math
import cv2
import os
import re

# file sort function
def fn(file):
  file_split = re.split('[-.]', os.path.basename(file))
  if file_split[1] == 'json':
    return int(file_split[0])
  else:
    return int(file_split[0]) * 100 + int(file_split[1])


# assemble cloud
def get_cloud(coords, colors):
  cloud = open3d.geometry.PointCloud()
  cloud.points = open3d.utility.Vector3dVector(coords)
  cloud.colors = open3d.utility.Vector3dVector(colors / 255.)
  return cloud


# chunk bounding box
def get_chunks(coords, size = 1):

  points = open3d.utility.Vector3dVector(coords)

  # bounding box
  eps = 1e-2
  bmin, bmax = coords.min(0), coords.max(0)
  ymin, ymax = bmin[1] - eps, bmax[1] + eps

  xmin, xmax = math.floor(bmin[0] - eps), math.floor(bmax[0] + eps)
  zmin, zmax = math.floor(bmin[2] - eps), math.floor(bmax[2] + eps)

  chunks = []

  for x in np.arange(xmin, xmax + 1, size):
    for z in np.arange(zmin, zmax + 1, size):
      chunk = open3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
        open3d.geometry.AxisAlignedBoundingBox(
          np.array([x, ymin, z]), np.array([x + size, ymax, z + size])
        )
      )
      idx = chunk.get_point_indices_within_bounding_box(points)
      if len(idx) > 0:
        chunks.append(chunk)

  return chunks


# stack from different cameras
def get_coords_and_colors(coord_files, color_files):

  # (n_points, 3) xyz
  coords = np.array([
    cv2.imread(coord_file, cv2.IMREAD_UNCHANGED)[:,:,:3]
    for coord_file in coord_files
  ]).reshape(-1, 3)

  # (n_points, 3) rgb
  colors = np.array([
    cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
    for color_file in color_files
  ]).reshape(-1, 3)

  return coords, colors