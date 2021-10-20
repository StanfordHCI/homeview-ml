from utils import get_cloud, get_chunks, fn
from glob import glob
import numpy as np
import torch
import math
import json
import cv2
import sys
import multiprocessing as mp


def is_empty(cloud):
  return np.asarray(cloud.points).shape[0] == 0


def diff_cloud(cloud0, cloud1):
  dist01 = cloud0.compute_point_cloud_distance(cloud1)
  dist10 = cloud1.compute_point_cloud_distance(cloud0)
  distance = max(np.asarray(dist01, dtype = np.float32).max(),# + \
                 np.asarray(dist10, dtype = np.float32).max())
  return distance
  # return 1 - math.exp(-distance)


def diff_rgb_cloud(cloud0, cloud1):
  p0 = np.concatenate([
      np.asarray(cloud0.points, dtype = np.float32),
      np.asarray(cloud0.colors, dtype = np.float32)
  ], axis = 1)
  p1 = np.concatenate([
      np.asarray(cloud1.points, dtype = np.float32),
      np.asarray(cloud1.colors, dtype = np.float32)
  ], axis = 1)

  def norm_min_0(p):
    return np.linalg.norm(p0 - p, axis = 1).min()

  def norm_min_1(p):
    return np.linalg.norm(p1 - p, axis = 1).min()

  d01 = [np.linalg.norm(p1 - p, axis = 1).min() for p in p0]
  d10 = [np.linalg.norm(p0 - p, axis = 1).min() for p in p1]
  
  # pool = mp.Pool()
  # d01 = pool.map(norm_min_0, p0)
  # d10 = pool.map(norm_min_1, p1)

  # d01 = -p0.repeat(p1.shape[0], axis = 0).reshape([p0.shape[0], p1.shape[0], p0.shape[1]])+p1
  # d10 = -p1.repeat(p0.shape[0], axis = 0).reshape([p1.shape[0], p0.shape[0], p1.shape[1]])+p0
  
  # distance = max(d01.max(), d10.max())
  distance = max(max(d01), max(d10))
  
  return distance


def diff_cloud_by_chunk(cloud0, cloud1):

  distances = []

  for chunk in chunks:
    chunk_cloud0 = cloud0.crop(chunk)
    chunk_cloud1 = cloud1.crop(chunk)
    if is_empty(chunk_cloud0) or is_empty(chunk_cloud1):
      distance = 2.0 * np.cbrt(chunk.volume())
    else:
      distance = diff_rgb_cloud(chunk_cloud0, chunk_cloud1)
    distances.append(distance)

  return torch.tensor(distances, dtype = torch.float32)




if __name__ == '__main__':

  # Step 1: get data files

  dataset_name = 'vh.' + sys.argv[1]
  coord_files = sorted(glob(dataset_name + '/*.exr'), key = fn)
  color_files = sorted(glob(dataset_name + '/*.png'), key = fn)
  json_files  = sorted(glob(dataset_name + '/*.json'), key = fn)

  n_data = len(coord_files)
  assert n_data == len(color_files)
  assert n_data == len(json_files)


  ### Step 2: read data

  # (n_data, n_points, 3) xyz
  coords  = [cv2.imread(coord_file, cv2.IMREAD_UNCHANGED).reshape(-1, 3) for coord_file in coord_files]

  # (n_data, n_points, 3) rgb
  colors  = [np.array(cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)).reshape(-1, 3) for color_file in color_files]

  # (n_data, n_sensors)
  sensors = [torch.tensor(
    [sensor['state'] for sensor in json.load(open(json_file))],
    dtype = torch.float32
  ) for json_file in json_files]


  ### Step 3: process data

  ref_cloud = get_cloud(coords[0], colors[0])
  dataset = []

  size = 1
  chunks = get_chunks(coords[0], size)

  # distances
  for i in range(0, n_data):
    print('processing %d/%d' % (i, n_data))
    cloud = get_cloud(coords[i], colors[i])
    distances = diff_cloud_by_chunk(ref_cloud, cloud)    # per chunk distance
    dataset.append((sensors[i], distances))


  ### Step 4: save
  torch.save(dataset[:100], dataset_name + '.train.pth')
  torch.save(dataset[100:], dataset_name + '.eval.pth')