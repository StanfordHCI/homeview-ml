from utils import get_cloud
from train import Model
import config
import torch

import numpy as np
import sys

import glob
import os

import open3d

import zmq
import json


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')

## load features to be matched once and for all
dataset_name = 'vh.' + sys.argv[1]
train_data = torch.load(dataset_name + '/train.pth')

features = np.array([data[1].tolist() for data in train_data])
## chunk features (n_frames, n_chunks * feature_dims) => (n_chunks, n_frames, feature_dims)
features = features.reshape(features.shape[0], -1, config.feature_dims).transpose(1, 0, 2)


## setup model
n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.feature_dims
model = Model(n_sensors, n_chunks, config.feature_dims)

ckpt = torch.load(dataset_name + '/model.pth')
model.load_state_dict(ckpt)
model.eval()


def get_frame_ids(sensors):
  pred_features = model(sensors).detach().numpy().reshape(-1, config.feature_dims)
  pred_frame_ids = [int(np.argmin(np.linalg.norm(feature - pred_feature, axis = 1))) \
    for feature, pred_feature in zip(features, pred_features)]
  return pred_frame_ids


prev_features = [[-1, -1]] * n_chunks
prev_frame_ids = [-1] * n_chunks
save_directory = dataset_name + '/tmp'

if not os.path.exists(save_directory):
  os.makedirs(save_directory)

while True:
  
  sensors = socket.recv()
  ## TODO: convert string/json to integer list [0, 1, 0, ..., 1]
  
  # sensors = np.random.randint(0, 2, n_sensors)
  # sensors = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  
  sensors = torch.Tensor(sensors)
  pred_frame_ids = get_frame_ids(sensors)

  files = glob.glob(save_directory + '/*')
  for file in files:
    os.remove(file)

  update_chunk_ids = []
  for chunk_id, (prev_frame_id, pred_frame_id, prev_feature, feature) in enumerate(
    zip(prev_frame_ids, pred_frame_ids, prev_features, features)):

    '''
      incremental update: the chunk is not updated if the matched feature does not
      change greatly(exceeds epsilon) compared to that of last request
    '''
    if pred_frame_id != prev_frame_id and np.linalg.norm(prev_feature - feature[pred_frame_id]) > config.epsilon:
  
      update_chunk_ids.append(chunk_id)
      prev_features[chunk_id] = feature[pred_frame_id]
      
      chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (pred_frame_id, chunk_id))['arr_0']
      
      ## assemble point cloud
      chunk_cloud = get_cloud(chunk_points[:, :3], chunk_points[:, 3:])

      ## local solution: save ply in a directory accessible by frontend
      open3d.io.write_point_cloud('%s/%d.ply' % (save_directory, chunk_id), chunk_cloud)

  prev_frame_ids = pred_frame_ids
  socket.send_string(json.dumps(update_chunk_ids))