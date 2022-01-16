from utils import get_coords_and_colors, get_chunks, get_file_lists
from train import Model
import config
import torch
import sys


## Step 1: get file lists storing point data and sensor data
dataset_name = config.dataset_prefix + '.' + sys.argv[1]
coord_files, color_files, _, n_frames = get_file_lists(dataset_name)


### Step 2: get chunk center positions
coords, _ = get_coords_and_colors(
  coord_files[config.ref_frame_id], 
  color_files[config.ref_frame_id]
)
chunks = get_chunks(coords, config.chunk_size)
chunk_positions = torch.Tensor([chunk.get_center() for chunk in chunks])


## Step 3: load and process model parameters

n_sensors = 11
n_chunks = 267
model = Model(n_sensors, n_chunks, config.feature_dims)

ckpt = torch.load(dataset_name + '/model.pth')
model.load_state_dict(ckpt)

weight, bias = list(model.parameters())
  
## combine geometry and luminance weight for each chunk
## (n_chunks * vector_dims, n_sensors) => (n_sensors, n_chunks, vector_dims) => (n_sensors, n_chunks)
weight = torch.transpose(weight, 1, 0).reshape((n_sensors, n_chunks, config.feature_dims)).sum(2)

sensor_positions = torch.mm(weight, chunk_positions) / weight.sum(1).reshape(-1, 1)
print(sensor_positions)