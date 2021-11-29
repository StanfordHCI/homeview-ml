import enum
from utils import get_cloud, get_chunks, get_coords_and_colors, fn
from glob import glob
import numpy as np
import config
import sys

import open3d

if __name__ == '__main__':

  # Step 1: get data file list

  # if config.n_cameras exceeds 100, modify fn in utils.py
  dataset_name = 'vh.' + sys.argv[1]

  coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key = fn)
  n_frames = len(coord_files) // config.n_cameras
  assert n_frames > config.n_train

  color_files = sorted(glob(dataset_name + '/raw/*.png'), key = fn)
  assert n_frames == len(color_files) // config.n_cameras


  # Step 2: read and process data

  # first frame as reference
  coords, colors = get_coords_and_colors(coord_files[:config.n_cameras], coord_files[:config.n_cameras])

  # bounding boxes
  chunks = get_chunks(coords, config.chunk_size)

  # Step 3: crop into chunks and save

  for frame_id in range(config.n_train):

    print('frame %d/%d' % (frame_id + 1, config.n_train))
    # (n_points, 3) xyz, (n_points, 3) rgb
    coords, colors = get_coords_and_colors(
      coord_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras],
      color_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras]
    )
    cloud = get_cloud(coords, colors)
    for chunk_id, chunk in enumerate(chunks):
      chunk_cloud = cloud.crop(chunk)
      coords = np.asarray(chunk_cloud.points)
      colors = np.asarray(chunk_cloud.colors)
      points = np.concatenate([coords, colors], axis = 1)
      np.savez_compressed(dataset_name + '/chunk/%d-%d.npz' % (frame_id, chunk_id), points)