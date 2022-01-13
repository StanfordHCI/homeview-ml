from train import Model, train
import config
import torch
import numpy as np
import sys

dataset_name = 'vh.' + sys.argv[1]

train_data = torch.load(dataset_name + '/train.pth')
eval_data = torch.load(dataset_name + '/eval.pth')

vecs = np.array([data[1].tolist() for data in train_data])
vecs = vecs.reshape(vecs.shape[0], -1, config.vector_dims).transpose(1, 0, 2)

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.vector_dims
model = Model(n_sensors, n_chunks, config.vector_dims)

ckpt = torch.load(dataset_name + '/model-4096-new-loss.pth')
model.load_state_dict(ckpt)
model.eval()

pcc = 0
n = 0
for sensors, gt_vecs in eval_data:
  gt_vecs = gt_vecs.reshape(-1, config.vector_dims)
  pred_vecs = model(sensors).detach().numpy().reshape(-1, config.vector_dims)
  for vec, pred_vec, gt_vec in zip(vecs, pred_vecs, gt_vecs):
    frame_id = np.argmin(np.linalg.norm(vec - pred_vec, axis = 1))
    vec = vec[frame_id]
    pcc += abs(vec[0] - gt_vec[0]) < 0.01 and abs(vec[1] - gt_vec[1]) < 0.05
    n += 1

print('ours:', pcc / n)

pcc = 0
n = 0
train_sensors = np.array([data[0].numpy() for data in train_data])

for sensors, gt_vecs in eval_data:
  gt_vecs = gt_vecs.reshape(-1, config.vector_dims)
  frame_id = np.argmin(np.linalg.norm(train_sensors - sensors.numpy(), axis = 1))
  print(frame_id)
  for vec, gt_vec in zip(vecs[frame_id], gt_vecs):
    pcc += abs(vec[0] - gt_vec[0]) < 0.01 and abs(vec[1] - gt_vec[1]) < 0.05
    n += 1

print('retrieve from history:', pcc / n)