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


# load vecs
dataset_name = "vh.cameras"
train_data = torch.load(dataset_name + '/train.pth')

vecs = np.array([data[1].tolist() for data in train_data])
# (n_chunks, n_frames, vector_dims) vector distances
vecs = vecs.reshape(vecs.shape[0], -1, config.vector_dims).transpose(1, 0, 2)

# instantiate model
n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.vector_dims
model = Model(n_sensors, n_chunks, config.vector_dims)

# load model
ckpt = torch.load(dataset_name + '/model-new-400-epoch.pth')
model.load_state_dict(ckpt)
model.eval()


def get_frame_ids(sensors):
    sensors = torch.tensor(sensors, dtype=torch.float32)
    pred_vecs = model(sensors).detach().numpy().reshape(-1, config.vector_dims)
    pred_frame_ids = [int(np.argmin(abs(vec - pred_vec))) for vec, pred_vec in zip(vecs, pred_vecs)]
    return pred_frame_ids


def my_write_point_cloud(new_frame_id, chunk_id):
    chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (new_frame_id, chunk_id))['arr_0']
    chunk_cloud = open3d.geometry.PointCloud()
    chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
    chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])
    address_zy = '../augmented-home-assistant-frontend/Assets/ply-common'
    open3d.io.write_point_cloud('%s/%d.ply' % (address_zy, chunk_id), chunk_cloud)


def process_request(recv_sensors, old_frame_ids, write_ply=False):
    """
    :param write_ply:
    :param recv_sensors: [0, 0, 0, 0, 0]
    :param old_frame_ids:
    :return:
    """
    # print("recv_sensors" + str(recv_sensors))
    # zhuoyue: these are the indices of all the lights, kind of hacky
    light_sensors_idx = [1, 2, 4, 6, 7]
    door_sensors_idx = [0, 3, 5, 8, 9]
    together_sensors_idx = light_sensors_idx + door_sensors_idx
    for i in range(len(recv_sensors)):
        sensors[together_sensors_idx[i] - 1] = recv_sensors[i]
    new_frame_ids = get_frame_ids(sensors)

    update_chunk_ids = []
    for chunk_id, (old_frame_id, new_frame_id) in enumerate(zip(old_frame_ids, new_frame_ids)):
        if new_frame_id != old_frame_id:
            print("Writing Chunk")
            print(chunk_id)
            update_chunk_ids.append(chunk_id)
            if write_ply:
                my_write_point_cloud(new_frame_id, chunk_id)

    return new_frame_ids, update_chunk_ids


if __name__ == '__main__':
    # Initial Sensor States
    sensors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    old_frame_ids = get_frame_ids(sensors)

    ##### Communication with Unity
    # context = zmq.Context()
    # socket = context.socket(zmq.REP)
    # socket.bind('tcp://*:5556')
    # while True:
    #     recv = socket.recv().decode('ascii')
    #     time.sleep(0.5)
    #     if recv == 'S':
    #         socket.send(b"Nothing")
    #     else:
    #         recv_sensors = [int(x) for x in recv.split(',')]
    #         old_frame_ids, update_chunk_ids = process_request(recv_sensors, old_frame_ids, write_ply=True)
    #         update_chunk_ids = '_'.join(str(id) for id in update_chunk_ids)
    #         socket.send_string(update_chunk_ids)

    ##### Test Stuff Locally
    recv_sensors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # recv_sensors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    old_frame_ids, update_chunk_ids = process_request(recv_sensors, old_frame_ids, write_ply=False)

""" Legacy
sensors = [sensor['state'] for sensor in json.load(open('./0.json'))]
"""
