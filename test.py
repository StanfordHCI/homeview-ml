from utils import get_cloud, get_chunks, get_coords_and_colors, get_file_lists
from train import Model
import numpy as np
import config
import open3d
import torch
import sys


## Step 1: get file lists storing point data and sensor data
dataset_name = config.dataset_prefix + '.' + sys.argv[1]
coord_files, color_files, _, n_frames = get_file_lists(dataset_name)


## Step 2: read and process data

train_data = torch.load(dataset_name + '/train.pth')
eval_data = torch.load(dataset_name + '/eval.pth')

features = np.array([data[1].tolist() for data in train_data])
## chunk features (n_frames, n_chunks * feature_dims) => (n_chunks, n_frames, feature_dims)
features = features.reshape(features.shape[0], -1, config.feature_dims).transpose(1, 0, 2)

## divide chunks from reference frame
coords, _ = get_coords_and_colors(
  coord_files[config.ref_frame_id], 
  color_files[config.ref_frame_id]
)
chunks = get_chunks(coords, config.chunk_size)


## Step 3: setup model

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.feature_dims
model = Model(n_sensors, n_chunks, config.feature_dims)

ckpt = torch.load(dataset_name + '/model.pth')
model.load_state_dict(ckpt)
model.eval()


## Step 4: test

test_id = int(sys.argv[2])
sensors, gt_features = eval_data[test_id - config.n_train] \
  if test_id >= config.n_train else train_data[test_id]
''' 
  if you modify the sensor data here for debuging, the ground_truth in step 6 
  will not be valid, since it is the ground truth of the unmodified one
'''
# sensors = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print('input:', sensors)
pred_features = model(sensors).detach().numpy().reshape(-1, config.feature_dims)


## Step 5: assemble, visualize

chunk_clouds = []

for chunk_id, (feature, pred_feature, chunk) in enumerate(zip(features, pred_features, chunks)):
  ## find the most matching frame by feature space distance
  frame_id = np.argmin(np.linalg.norm(feature - pred_feature, axis = 1))
  
  ## debug
  # if chunk_id == 100:
  #   print('ground truth:', features[chunk_id * 2:chunk_id * 2 + 2])
  #   print('prediction:', pred_features)
  #   print('matched frame:', frame_id, features[frame_id])

  '''
    load chunk points from binary npz files(gen by localized.py), this is much faster than 
    calling get_coords_and_colors(), which needs to decode exr and png camera images.
  '''
  chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (frame_id, chunk_id))['arr_0']
  
  '''
    The following commented code is used to produce visual horizontal gaps between chunks for illustration only.
    This is made by applying a small x/z shift to all points towards the center
  '''
  # chunk_center = chunk.get_center()
  # chunk_points[:, [0,2]] -= 0.15 * (chunk_points[:, [0,2]] - chunk_center[[0,2]])

  ## assemble point cloud
  chunk_cloud = get_cloud(chunk_points[:, :3], chunk_points[:, 3:])
  chunk_clouds.append(chunk_cloud)


### Step 6: remove ceiling

## ground truth
coords, colors = get_coords_and_colors(coord_files[test_id], color_files[test_id])

## creates heavily noised point cloud for illustrating synthetic method only
# coords = coords + np.random.uniform(-0.2, 0.2, size = coords.shape)

ground_truth = get_cloud(coords, colors)
ceiling = ground_truth.get_axis_aligned_bounding_box()

no_ceiling = open3d.geometry.AxisAlignedBoundingBox(
  np.array(ceiling.min_bound),
  np.array([ceiling.max_bound[0], ceiling.max_bound[1] - 1e-1, ceiling.max_bound[2]])
)

ground_truth = ground_truth.crop(no_ceiling)
chunk_clouds = [cloud.crop(no_ceiling) for cloud in chunk_clouds]

open3d.visualization.draw_geometries([ground_truth], width = 1080, height = 1080, lookat = np.array([0, 0, -3.5]), up = np.array([0, 0, -1]), front = np.array([0, 1, 0]), zoom = 0.6)
open3d.visualization.draw_geometries(chunk_clouds, width = 1080, height = 1080, lookat = np.array([0, 0, -3.5]), up = np.array([0, 0, -1]), front = np.array([0, 1, 0]), zoom = 0.6)
## visualize a subset of chunk point clouds
# open3d.visualization.draw_geometries(chunk_clouds[163:164])