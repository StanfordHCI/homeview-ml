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

# load vecs
dataset_name = 'vh.' + sys.argv[1]
train_data = torch.load(dataset_name + '/train.pth')

vecs = np.array([data[1].tolist() for data in train_data])
# (n_chunks, n_frames, vector_dims) vector distances
vecs = vecs.reshape(vecs.shape[0], -1, config.vector_dims).transpose(1, 0, 2)

# instantiate model
n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.vector_dims
model = Model(n_sensors, n_chunks, config.vector_dims)

# load model
ckpt = torch.load(dataset_name + '/.pth')
model.load_state_dict(ckpt)
model.eval()


def get_frame_ids(sensors):
  pred_vecs = model(sensors).detach().numpy().reshape(-1, config.vector_dims)
  pred_frame_ids = [int(np.argmin(np.linalg.norm(vec - pred_vec, axis = 1))) \
    for vec, pred_vec in zip(vecs, pred_vecs)]
  return pred_frame_ids


prev_vecs = [[-1, -1]] * n_chunks
prev_frame_ids = [-1] * n_chunks
save_directory = dataset_name + '/tmp'

if not os.path.exists(save_directory):
  os.makedirs(save_directory)

while True:
  # TODO: convert string/json to integer list [0, 1, 0, ..., 1]
  sensors = socket.recv()
  # sensors = np.random.randint(0, 2, n_sensors)
  # sensors = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  sensors = torch.Tensor(sensors)

  pred_frame_ids = get_frame_ids(sensors)

  files = glob.glob(save_directory + '/*')
  for file in files:
    os.remove(file)

  update_chunk_ids = []
  for chunk_id, (prev_frame_id, pred_frame_id, prev_vec, vec) in enumerate(
    zip(prev_frame_ids, pred_frame_ids, prev_vecs, vecs)):

    if pred_frame_id != prev_frame_id and np.linalg.norm(prev_vec - vec[pred_frame_id]) > config.epsilon:

      update_chunk_ids.append(chunk_id)
      prev_vecs[chunk_id] = vec[pred_frame_id]
      
      chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (pred_frame_id, chunk_id))['arr_0']
      
      chunk_cloud = open3d.geometry.PointCloud()
      chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
      chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])
      
      open3d.io.write_point_cloud('%s/%d.ply' % (save_directory, chunk_id), chunk_cloud)

  print(len(update_chunk_ids))
  prev_frame_ids = pred_frame_ids
  socket.send_string(json.dumps(update_chunk_ids))