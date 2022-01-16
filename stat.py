from train import Model, train
import numpy as np
import config
import torch
import sys

dataset_name = config.dataset_prefix + '.' + sys.argv[1]

train_data = torch.load(dataset_name + '/train.pth')
eval_data = torch.load(dataset_name + '/eval.pth')

features = np.array([data[1].tolist() for data in train_data])
## chunk features (n_frames, n_chunks * feature_dims) => (n_chunks, n_frames, feature_dims)
features = features.reshape(features.shape[0], -1, config.feature_dims).transpose(1, 0, 2)

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.feature_dims
model = Model(n_sensors, n_chunks, config.feature_dims)

ckpt = torch.load(dataset_name + '/model.pth')
model.load_state_dict(ckpt)
model.eval()


geometry_threshold = 0.01
illuminance_threshold = 0.05

pcc = 0
n = 0
for sensors, gt_features in eval_data:
  gt_features = gt_features.reshape(-1, config.feature_dims)
  pred_features = model(sensors).detach().numpy().reshape(-1, config.feature_dims)
  for feature, pred_feature, gt_feature in zip(features, pred_features, gt_features):
    frame_id = np.argmin(np.linalg.norm(feature - pred_feature, axis = 1))
    feature = feature[frame_id]
    pcc += abs(feature[0] - gt_feature[0]) < geometry_threshold and abs(feature[1] - gt_feature[1]) < illuminance_threshold
    n += 1

print('ours:', pcc / n)

pcc = 0
n = 0
train_sensors = np.array([data[0].numpy() for data in train_data])

## retrieve-from-histroy matches by sensor data
for sensors, gt_features in eval_data:
  gt_features = gt_features.reshape(-1, config.feature_dims)
  frame_id = np.argmin(np.linalg.norm(train_sensors - sensors.numpy(), axis = 1))
  print(frame_id)
  for feature, gt_feature in zip(features[frame_id], gt_features):
    pcc += abs(feature[0] - gt_feature[0]) < geometry_threshold and abs(feature[1] - gt_feature[1]) < illuminance_threshold
    n += 1

print('retrieve from history:', pcc / n)