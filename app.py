from train import Model
import torch

from flask import Flask
from flask_compress import Compress
from flask.helpers import make_response

import numpy as np
import sys

import base64


# load vecs
dataset_name = 'vh.' + sys.argv[1]
train_data = torch.load(dataset_name + '/train.pth')

# (n_chunks, n_frames) distances
vecs = np.array([data[1].tolist() for data in train_data]).transpose()


# instantiate model
n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0]
model = Model(n_sensors, n_chunks)
# load model
ckpt = torch.load(dataset_name + '/.pth')
model.load_state_dict(ckpt)
model.eval()


def get_frame_ids(sensors):
  pred_vecs = model(sensors).detach().numpy()
  pred_frame_ids = [np.argmin(abs(vec - pred_vec)) \
    for vec, pred_vec in zip(vecs, pred_vecs)]
  return pred_frame_ids


server = Flask(__name__)
server.config.update(
  ENV = 'development',
  DEBUG = True
)
Compress(server)

def static_vars(**kwargs):
  def decorate(func):
    for k in kwargs:
      setattr(func, k, kwargs[k])
    return func
  return decorate

@static_vars(prev_frame_ids = [0] * n_chunks)
@server.route('/', methods = ['GET'])
def getCloud():

  sensors = train_data[np.random.randint(100)][0]
  pred_frame_ids = get_frame_ids(sensors)

  response = {}
  for chunk_id, (prev_frame_id, pred_frame_id) in enumerate(zip(getCloud.prev_frame_ids, pred_frame_ids)):
    if pred_frame_id != prev_frame_id:
      chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (pred_frame_id, chunk_id))['arr_0']
      response[chunk_id] = chunk_points.tobytes()

  getCloud.prev_frame_ids = pred_frame_ids
  print(len(response.keys()) + ' chunks updated')

  return response

server.run(debug = True)