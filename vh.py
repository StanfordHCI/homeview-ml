from utils import get_cloud, get_chunks, get_coords_and_colors, fn
from glob import glob
import numpy as np
import config
import torch
import json
import sys
import multiprocessing as mp


def is_empty(cloud):
  return np.asarray(cloud.points).shape[0] == 0


def diff_geo(cloud0, cloud1):
  dist01 = cloud0.compute_point_cloud_distance(cloud1)
  dist10 = cloud1.compute_point_cloud_distance(cloud0)

  # dist01 = np.asarray(dist01, dtype = int)
  # dist10 = np.asarray(dist10, dtype = int)
  # points0 = np.asarray(cloud0.points)
  # points1 = np.asarray(cloud1.points)
  # colors0 = np.asarray(cloud0.colors)
  # colors1 = np.asarray(cloud1.colors)
  
  # dist01 = np.concatenate([points0 - points1[dist01], colors0 - colors1[dist01]], axis = 1)
  # dist01 = np.linalg.norm(dist01, axis = 1)
  
  # dist10 = np.concatenate([points1 - points0[dist10], colors1 - colors0[dist10]], axis = 1)
  # dist10 = np.linalg.norm(dist10, axis = 1)

  distance = max(np.asarray(dist01, dtype = np.float32).max(),# + \
                 np.asarray(dist10, dtype = np.float32).max())
  return distance
  # return 1 - math.exp(-distance)


def diff_vec(cloud0, cloud1):
  geo_distance = diff_geo(cloud0, cloud1)
  illuminance = (np.asarray(cloud1.colors, dtype = np.float32) * [0.3, 0.6, 0.1]).sum(1).mean()
  return [geo_distance, illuminance]


def diff_cloud_by_chunk(cloud0, cloud1, chunks):

  distances = []

  for chunk in chunks:
    chunk_cloud0 = cloud0.crop(chunk)
    chunk_cloud1 = cloud1.crop(chunk)
    if is_empty(chunk_cloud0) or is_empty(chunk_cloud1):
      distance = [2.0 * np.cbrt(chunk.volume()), -1.0]
    else:
      distance = diff_vec(chunk_cloud0, chunk_cloud1)
    distances.extend(distance)

  return torch.tensor(distances, dtype = torch.float32)



if __name__ == '__main__':

  # Step 1: get data file list

  # if config.n_cameras exceeds 100, modify fn in utils.py
  dataset_name = 'vh.' + sys.argv[1]

  coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key = fn)
  n_frames = len(coord_files) // config.n_cameras
  assert n_frames > config.n_train

  color_files = sorted(glob(dataset_name + '/raw/*.png'), key = fn)
  assert n_frames == len(color_files) // config.n_cameras

  json_files = sorted(glob(dataset_name + '/raw/*.json'), key = fn)
  json_files = json_files[0:n_frames]
  # assert n_frames == len(json_files)


  # Step 2: read and process data

  # (n_frames, n_sensors)
  sensors = [torch.tensor(
    [sensor['state'] for sensor in json.load(open(json_file))],
    dtype = torch.float32
  ) for json_file in json_files]

  # first frame as reference
  coords, colors = get_coords_and_colors(coord_files[:config.n_cameras], color_files[:config.n_cameras])
  ref_cloud = get_cloud(coords, colors)

  # bounding boxes
  chunks = get_chunks(coords, config.chunk_size)

  dataset = []

  for frame_id in range(0, n_frames):

    print('processing %d/%d' % (frame_id + 1, n_frames))
    # (n_points, 3) xyz, (n_points, 3) rgb
    coords, colors = get_coords_and_colors(
      coord_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras],
      color_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras]
    )
    cloud = get_cloud(coords, colors)
    # per chunk distance
    distances = diff_cloud_by_chunk(ref_cloud, cloud, chunks)
    dataset.append((sensors[frame_id], distances))


  ### Step 3: save
  torch.save(dataset[:config.n_train], dataset_name + '/train.pth')
  torch.save(dataset[config.n_train:], dataset_name + '/eval.pth')