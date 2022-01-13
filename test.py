from utils import get_cloud, get_chunks, get_coords_and_colors, fn
from train import Model
from glob import glob
import numpy as np
import config
import open3d
import torch
import sys


### Step 1: read data, get data file list

dataset_name = 'vh.' + sys.argv[1]

## if config.n_cameras exceeds 100, modify fn in utils.py

coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key = fn)
n_frames = len(coord_files) // config.n_cameras
assert n_frames > config.n_train

color_files = sorted(glob(dataset_name + '/raw/*.png'), key = fn)
assert n_frames == len(color_files) // config.n_cameras



### Step 2: process data

train_data = torch.load(dataset_name + '/train.pth')
eval_data = torch.load(dataset_name + '/eval.pth')

vecs = np.array([data[1].tolist() for data in train_data])
## (n_chunks, n_frames, vector_dims) vector distances
vecs = vecs.reshape(vecs.shape[0], -1, config.vector_dims).transpose(1, 0, 2)

## bounding boxes
coords, _ = get_coords_and_colors(coord_files[:config.n_cameras], color_files[:config.n_cameras])
chunks = get_chunks(coords, config.chunk_size)



### Step 3: setup model

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.vector_dims
model = Model(n_sensors, n_chunks, config.vector_dims)

ckpt = torch.load(dataset_name + '/model-4096-new-loss.pth')
model.load_state_dict(ckpt)
model.eval()


### Step 4: evaluation

test_id = int(sys.argv[2])
sensors, gt_vecs = eval_data[test_id - config.n_train] \
  if test_id >= config.n_train else train_data[test_id]
## if you modify the sensor data here for debuging, the ground_truth in step 6
## will not be valid since it is the ground truth of the unmodified one
# sensors = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# sensors = torch.Tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
print(sensors)
pred_vecs = model(sensors).detach().numpy().reshape(-1, config.vector_dims)



### Step 5: assemble, visualize

frame_clouds = {}
chunk_clouds = []

assemble_time = 0.0

for chunk_id, (vec, pred_vec, chunk) in enumerate(zip(vecs, pred_vecs, chunks)):
  ## find closest distance (most matching frame) from database
  frame_id = np.argmin(np.linalg.norm(vec - pred_vec, axis = 1))
  if chunk_id == 100:
    for data, v in zip(train_data, vec):
      print(data[0][[3, 7]].tolist(), v)
    print('ground truth: ', gt_vecs[chunk_id * 2:chunk_id * 2 + 2])
    print('prediction: ', pred_vec)
    print('selection: frame ', frame_id, vec[frame_id])

  chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (frame_id, chunk_id))['arr_0']
  chunk_cloud = open3d.geometry.PointCloud()
  # chunk_center = chunk.get_center()
  # chunk_points[:, [0,2]] -= 0.15 * (chunk_points[:, [0,2]] - chunk_center[[0,2]])
  chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
  chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])

  # chunk_cloud = frame_clouds[frame_id].crop(chunk)
  chunk_clouds.append(chunk_cloud)



### Step 6: remove ceiling

coords, colors = get_coords_and_colors(
  coord_files[test_id * config.n_cameras:(test_id + 1) * config.n_cameras],
  color_files[test_id * config.n_cameras:(test_id + 1) * config.n_cameras]
)
# coords = coords + np.random.uniform(-0.2, 0.2, size = coords.shape)
ground_truth = get_cloud(coords, colors)

ceiling = ground_truth.get_axis_aligned_bounding_box()

no_ceiling = open3d.geometry.AxisAlignedBoundingBox(
  np.array(ceiling.min_bound),
  np.array([ceiling.max_bound[0], ceiling.max_bound[1] - 1e-1, ceiling.max_bound[2]])
) ## any more elegant way to 'crop' a bounding box?

ground_truth = ground_truth.crop(no_ceiling)
chunk_clouds = [cloud.crop(no_ceiling) for cloud in chunk_clouds]

open3d.visualization.draw_geometries([ground_truth], width = 1080, height = 1080, lookat = np.array([0, 0, -3.5]), up = np.array([0, 0, -1]), front = np.array([0, 1, 0]), zoom = 0.6)
open3d.visualization.draw_geometries(chunk_clouds, width = 1080, height = 1080, lookat = np.array([0, 0, -3.5]), up = np.array([0, 0, -1]), front = np.array([0, 1, 0]), zoom = 0.6)
# open3d.visualization.draw_geometries(chunk_clouds[163:164])