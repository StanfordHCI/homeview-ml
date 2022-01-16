from utils import get_cloud, get_chunks, get_coords_and_colors, get_file_lists
import numpy as np
import config
import torch
import json
import sys


def is_empty(cloud):
  return np.asarray(cloud.points).shape[0] == 0

## computes the geometry distance between two point clouds, a modified symmetric Chamfer Distance
def compute_geometry_distance(cloud0, cloud1):
  dist01 = cloud0.compute_point_cloud_distance(cloud1)
  dist10 = cloud1.compute_point_cloud_distance(cloud0)
  distance = max(np.asarray(dist01, dtype = np.float32).max(),# + \
                 np.asarray(dist10, dtype = np.float32).max())
  return distance
  # return 1 - math.exp(-distance)


## computes the feature(geometry + illuminance) of point cloud
def compute_feature(base_cloud, cloud):
  # geometry distance
  geo_distance = compute_geometry_distance(base_cloud, cloud)
  # illuminace is approximated the Y channel of YUV converted from RGB
  illuminance = (np.asarray(cloud.colors, dtype = np.float32) * [0.3, 0.6, 0.1]).sum(1).mean()
  return [geo_distance, illuminance]


## compute the features of point cloud in a chunk wise manner
def compute_chunk_features(cloud):

  features = []

  for chunk in chunks:
    chunk_ref_cloud = ref_cloud.crop(chunk)
    chunk_cloud = cloud.crop(chunk)
    if is_empty(chunk_ref_cloud) or is_empty(chunk_cloud):
      feature = [2.0 * np.cbrt(chunk.volume()), -1.0]
    else:
      feature = compute_feature(chunk_ref_cloud, chunk_cloud)
    features.extend(feature)

  return torch.tensor(features, dtype = torch.float32)



if __name__ == '__main__':

  ## Step 1: get file lists storing point data and sensor data
  dataset_name = config.dataset_prefix + '.' + sys.argv[1]
  coord_files, color_files, sensor_files, n_frames = get_file_lists(dataset_name)

  ## Step 2: read and process data
  ## (n_frames, n_sensors)
  sensors = [torch.tensor(
    [sensor['state'] for sensor in json.load(open(sensor_file))],
    dtype = torch.float32
  ) for sensor_file in sensor_files]

  ## reference frame
  coords, colors = get_coords_and_colors(
    coord_files[config.ref_frame_id], 
    color_files[config.ref_frame_id]
  )
  ref_cloud = get_cloud(coords, colors / 255.0)

  ## divide chunks
  chunks = get_chunks(coords, config.chunk_size)

  dataset = []

  for frame_id in range(1, n_frames):

    print('processing %d/%d' % (frame_id + 1, n_frames))
    ## (n_points, 3) xyz, (n_points, 3) rgb
    coords, colors = get_coords_and_colors(
      coord_files[frame_id],
      color_files[frame_id]
    )
    cloud = get_cloud(coords, colors / 255.0)
     
    features = compute_chunk_features(cloud)
    dataset.append((sensors[frame_id], features))


  ## Step 3: save
  torch.save(dataset[:config.n_train], dataset_name + '/train.pth')
  torch.save(dataset[config.n_train:], dataset_name + '/eval.pth')