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

# if config.n_cameras exceeds 100, modify fn in utils.py

coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key = fn)
n_frames = len(coord_files) // config.n_cameras
assert n_frames > config.n_train

color_files = sorted(glob(dataset_name + '/raw/*.png'), key = fn)
assert n_frames == len(color_files) // config.n_cameras

json_files = sorted(glob(dataset_name + '/raw/*.json'), key = fn)
json_files = json_files[0:n_frames]
# assert n_frames == len(json_files)



### Step 2: process data

train_data = torch.load(dataset_name + '/train.pth')
eval_data = torch.load(dataset_name + '/eval.pth')

# (n_chunks, n_frames) distances
diffs = np.array([data[1].tolist() for data in train_data]).transpose()

# bounding boxes
coords, _ = get_coords_and_colors(coord_files[:config.n_cameras], coord_files[:config.n_cameras])
chunks = get_chunks(coords, config.chunk_size)



### Step 3: setup model

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0]
model = Model(n_sensors, n_chunks)

ckpt = torch.load(dataset_name + '/.pth')
model.load_state_dict(ckpt)
model.eval()



### Step 4: evaluation

test_id = int(sys.argv[2])
sensors = eval_data[test_id - config.n_train][0] \
  if test_id > config.n_train else train_data[test_id][0]

predicts = model(sensors).detach().numpy()


### Step 5: assemble, visualize

frame_clouds = {}
chunk_clouds = []

for chunk_id, (diff, predict, chunk) in enumerate(zip(diffs, predicts, chunks)):
  # find closest distance (most matching frame) from database
  frame_id = np.argmin(abs(diff - predict))

  if frame_id not in frame_clouds:
    coords, colors = get_coords_and_colors(
      coord_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras],
      color_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras]
    )
    frame_clouds[frame_id] = get_cloud(coords, colors)

  chunk_cloud = frame_clouds[frame_id].crop(chunk)
  chunk_clouds.append(chunk_cloud)
  # DEBUG
  # if chunk_id == 51:
  #   print(predict, frame_id)
  #   print(diff)
  #   open3d.visualization.draw_geometries([chunk_clouds[frame_id].crop(chunk)])


### Step 6: remove ceiling

coords, colors = get_coords_and_colors(
  coord_files[test_id * config.n_cameras:(test_id + 1) * config.n_cameras],
  color_files[test_id * config.n_cameras:(test_id + 1) * config.n_cameras]
)
ground_truth = get_cloud(coords, colors)

ceiling = ground_truth.get_axis_aligned_bounding_box()

no_ceiling = open3d.geometry.AxisAlignedBoundingBox(
  np.array(ceiling.min_bound), 
  np.array([ceiling.max_bound[0], ceiling.max_bound[1] - 1e-1, ceiling.max_bound[2]])
) # any more elegant way to 'crop' a bounding box?

ground_truth = ground_truth.crop(no_ceiling)
chunk_clouds = [cloud.crop(no_ceiling) for cloud in chunk_clouds]

open3d.visualization.draw_geometries([ground_truth])
open3d.visualization.draw_geometries(chunk_clouds)