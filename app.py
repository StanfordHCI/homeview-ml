from train import Model
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
socket.bind('tcp://*:5556')

# load vecs
# dataset_name = 'vh.' + sys.argv[1]
dataset_name = "vh.cameras"
train_data = torch.load(dataset_name + '/train.pth')

# (n_chunks, n_frames) distances
vecs = np.array([data[1].tolist() for data in train_data]).transpose()

# instantiate model
n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0]
model = Model(n_sensors, n_chunks)

# load model
ckpt = torch.load(dataset_name + '/t.pth')
model.load_state_dict(ckpt)
model.eval()


def get_frame_ids(sensors):
    pred_vecs = model(sensors).detach().numpy()
    pred_frame_ids = [int(np.argmin(abs(vec - pred_vec))) \
                      for vec, pred_vec in zip(vecs, pred_vecs)]
    return pred_frame_ids


prev_frame_ids = [-1] * n_chunks
save_directory = dataset_name + '/tmp'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

sensors = [sensor['state'] for sensor in json.load(open('./99.json'))][0:88]

while True:
    rec_sensors = socket.recv()
    print(rec_sensors)
    rec_sensors = rec_sensors.decode('ascii')
    print(rec_sensors)
    import time

    time.sleep(1)
    if rec_sensors == 'S':
        socket.send(b"Nothing")
    else:
        rec_sensors = [int(x) for x in rec_sensors.split(',')]
        # rec_sensors = [0, 0, 0, 0, 0]
        # zhuoyue: these are the indices of all the lights, kind of hacky
        print(rec_sensors)
        light_sensors_idx = [12,
                             40,
                             44,
                             62,
                             75]
        for i in range(len(rec_sensors)):
            sensors[light_sensors_idx[i] - 1] = rec_sensors[i]
        sensors = torch.tensor(sensors, dtype=torch.float32)

        pred_frame_ids = get_frame_ids(sensors)

        files = glob.glob(save_directory + '/*')
        for file in files:
            os.remove(file)

        for chunk_id, (prev_frame_id, pred_frame_id) in enumerate(zip(prev_frame_ids, pred_frame_ids)):
            if pred_frame_id != prev_frame_id:
                chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (pred_frame_id, chunk_id))['arr_0']
                chunk_cloud = open3d.geometry.PointCloud()
                chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
                chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])
                address_zy = '/Users/zhuoyuelyu/Documents/a-Stanford/StanfordHCI/virtualhome-11/Assets/ply-common'
                open3d.io.write_point_cloud('%s/%d.ply' % (address_zy, chunk_id), chunk_cloud)

        prev_frame_ids = pred_frame_ids
        # prev_frame_ids = [1, 2, 3]
        # convert it into string
        prev_frame_ids = '_'.join(str(id) for id in prev_frame_ids)
        socket.send_string(prev_frame_ids)
