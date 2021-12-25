from utils import get_cloud, get_chunks, get_coords_and_colors, merge_clouds, fn
from glob import glob
import numpy as np
import config
import open3d
import torch
import time
import sys


class Model(torch.nn.Module):
    def __init__(self, sensors, chunks, vector_dims):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(sensors, chunks * vector_dims),
            torch.nn.ReLU(),
        )

    def forward(self, input):
        output = self.linear(input)
        return output


# open3d.visualization.webrtc_server.enable_webrtc()
### Step 1: read data, get data file list

dataset_name = "vh.cameras"

### Step 2: process data
train_data = torch.load(dataset_name + '/train-4-sensors.pth')
eval_data = torch.load(dataset_name + '/eval-4-sensors.pth')

vecs = np.array([data[1].tolist() for data in train_data])
vecs = vecs.reshape(vecs.shape[0], -1, config.vector_dims).transpose(1, 0, 2)

### Step 3: setup model

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.vector_dims
model = Model(n_sensors, n_chunks, config.vector_dims)

ckpt = torch.load(dataset_name + '/model-new-4-sensors-2000-adam-lr-4-l2-3-not-0-1.pth')
model.load_state_dict(ckpt)
model.eval()

### Step 4: evaluation

def render_home(sensors):
    print(sensors)
    sensors = torch.tensor(sensors, dtype=torch.float32)
    pred_vecs = model(sensors).detach().numpy().reshape(-1, config.vector_dims)
    ### Step 5: assemble, visualize

    assemble_start = time.time()
    print('assemble start')
    frame_clouds = {}
    chunk_clouds = []

    assemble_time = 0.0
    # open3d.visualization.webrtc_server.enable_webrtc()
    for chunk_id, (vec, pred_vec) in enumerate(zip(vecs, pred_vecs)):
        # find closest distance (most matching frame) from database
        frame_id = np.argmin(np.linalg.norm(vec - pred_vec, axis=1))
        chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (frame_id, chunk_id))['arr_0']
        chunk_cloud = open3d.geometry.PointCloud()
        chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
        chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])

        # chunk_cloud = frame_clouds[frame_id].crop(chunk)
        chunk_clouds.append(chunk_cloud)

    chunk_clouds = merge_clouds(chunk_clouds)
    # open3d.visualization.draw_geometries([chunk_clouds])
    return chunk_clouds