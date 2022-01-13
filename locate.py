from glob import glob
import sys

from utils import get_coords_and_colors, get_chunks, fn
from train import Model
import config
import torch


### Step 1: read data, get data file list

dataset_name = 'vh.' + sys.argv[1]

# if config.n_cameras exceeds 100, modify fn in utils.py

coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key = fn)
n_frames = len(coord_files) // config.n_cameras
assert n_frames > config.n_train

color_files = sorted(glob(dataset_name + '/raw/*.png'), key = fn)
assert n_frames == len(color_files) // config.n_cameras



### Step 2: get chunk center positions

coords, _ = get_coords_and_colors(coord_files[:config.n_cameras], color_files[:config.n_cameras])
chunks = get_chunks(coords, config.chunk_size)

chunk_positions = torch.Tensor([chunk.get_center() for chunk in chunks])



### Step 3: load and process model parameters

n_sensors = 11
n_chunks = 267
model = Model(n_sensors, n_chunks, config.vector_dims)

ckpt = torch.load(dataset_name + '/model-4096-new-loss.pth')
model.load_state_dict(ckpt)

weight, bias = list(model.parameters())
  
# combine geometry and luminance weight for each chunk
# (n_chunks * vector_dims, n_sensors) => (n_sensors, n_chunks, vector_dims) => (n_sensors, n_chunks)
weight = torch.transpose(weight, 1, 0).reshape((n_sensors, n_chunks, config.vector_dims)).sum(2)

sensor_positions = torch.mm(weight, chunk_positions) / weight.sum(1).reshape(-1, 1)
print(sensor_positions)