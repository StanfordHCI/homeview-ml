from utils import get_chunks, get_cloud, fn
from train import Model

from glob import glob
import numpy as np
import open3d
import torch
import json
import cv2
import sys


### Step 1: read data

dataset = 'vh.' + sys.argv[1]

train_data = torch.load(dataset + '.train.pth')
eval_data = torch.load(dataset + '.eval.pth')

n_train_data = len(train_data)

coord_files = sorted(glob(dataset + '/*.exr'), key = fn)
color_files = sorted(glob(dataset + '/*.png'), key = fn)
json_files = sorted(glob(dataset + '/*.json'), key = fn)

assert len(coord_files) == len(color_files)
assert len(coord_files) == len(json_files)

# (n_data, n_points, 3) xyz
coords  = [cv2.imread(coord_file, cv2.IMREAD_UNCHANGED).reshape(-1, 3) for coord_file in coord_files]

# (n_data, n_points, 3) rgb
colors  = [np.array(cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)).reshape(-1, 3) for color_file in color_files]

# (n_data, n_sensors)
sensors = np.array([
  [sensor['state'] for sensor in json.load(open(json_file))]
for json_file in json_files])



### Step 2: process data

# use training data as database
clouds = [get_cloud(coords[i], colors[i]) for i in range(n_train_data)]

# (n_chunks, n_data) distances
diffs = np.transpose(np.array([data[1].tolist() for data in train_data]))

size = 1
chunks = get_chunks(coords[0], size)



### Step 3: setup model

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0]
model = Model(n_sensors, n_chunks)

ckpt = torch.load(dataset + '.pth')
model.load_state_dict(ckpt)
model.eval()



### Step 4: evaluation

eval_id = int(sys.argv[2])
sensors = eval_data[eval_id][0]

predicts = model(sensors).detach().numpy()


### Step 5: assemble, visualize

chunk_clouds = []

for chunk_id, (diff, predict, chunk) in enumerate(zip(diffs, predicts, chunks)):
  # find closest distance (most matching chunk) from database
  frame_id = np.argmin(abs(diff - predict))
  chunk_cloud = clouds[frame_id].crop(chunk)
  chunk_clouds.append(chunk_cloud)
  # DEBUG
  # if chunk_id == 51:
  #   print(predict, frame_id)
  #   print(diff)
  #   open3d.visualization.draw_geometries([clouds[frame_id].crop(chunk)])


open3d.visualization.draw_geometries(chunk_clouds)