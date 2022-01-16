from utils import get_cloud, get_chunks, get_coords_and_colors, get_file_lists
import numpy as np
import config
import open3d
import sys


if __name__ == '__main__':

  ## Step 1: get file lists storing point data and sensor data
  dataset_name = config.dataset_prefix + '.' + sys.argv[1]
  coord_files, color_files, _, n_frames = get_file_lists(dataset_name)


  ## Step 2: read and process data

  ## divide chunks from reference frame
  coords, _ = get_coords_and_colors(
    coord_files[config.ref_frame_id], 
    color_files[config.ref_frame_id]
  )
  chunks = get_chunks(coords, config.chunk_size)


  ## Step 3: crop by chunk and save compression
  for frame_id in range(config.n_train):

    print('frame %d/%d' % (frame_id + 1, config.n_train))
    # (n_points, 3) xyz, (n_points, 3) rgb
    coords, colors = get_coords_and_colors(
      coord_files[frame_id],
      color_files[frame_id]
    )
    cloud = get_cloud(coords, colors / 255.0)
    for chunk_id, chunk in enumerate(chunks):
      chunk_cloud = cloud.crop(chunk)
      coords = np.asarray(chunk_cloud.points)
      colors = np.asarray(chunk_cloud.colors)
      points = np.concatenate([coords, colors], axis = 1)
      np.savez_compressed(dataset_name + '/chunk/%d-%d.npz' % (frame_id, chunk_id), points)