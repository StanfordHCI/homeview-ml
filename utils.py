import numpy as np
import open3d
import math
import os
import re

# file sort function
def fn(file):
  return int(re.split('[-.]', os.path.basename(file))[0])

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