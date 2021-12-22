from utils import get_cloud, get_chunks, get_coords_and_colors, merge_clouds, fn
from train import Model
from glob import glob
import numpy as np
import config
import open3d
import torch
import time
import sys

### Step 1: read data, get data file list

dataset_name = 'vh.' + sys.argv[1]

# if config.n_cameras exceeds 100, modify fn in utils.py

coord_files = sorted(glob(dataset_name + '/raw/*.exr'), key=fn)
n_frames = len(coord_files) // config.n_cameras
assert n_frames > config.n_train

color_files = sorted(glob(dataset_name + '/raw/*.png'), key=fn)
assert n_frames == len(color_files) // config.n_cameras

json_files = sorted(glob(dataset_name + '/raw/*.json'), key=fn)
json_files = json_files[0:n_frames]
# assert n_frames == len(json_files)


### Step 2: process data

train_data = torch.load(dataset_name + '/train.pth')
eval_data = torch.load(dataset_name + '/eval.pth')

# (n_chunks, n_frames) distances
vecs = np.array([data[1].tolist() for data in train_data])
vecs = vecs.reshape(vecs.shape[0], -1, config.vector_dims).transpose(1, 0, 2)

# bounding boxes
coords, _ = get_coords_and_colors(coord_files[:config.n_cameras], coord_files[:config.n_cameras])
chunks = get_chunks(coords, config.chunk_size)

# frame_clouds = []

# for frame_id in range(n_frames):
#   coords, colors = get_coords_and_colors(
#     coord_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras],
#     color_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras]
#   )
#   frame_clouds.append(get_cloud(coords, colors))
# frame_clouds[frame_id] = get_cloud(coords, colors)


### Step 3: setup model

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0] // config.vector_dims
model = Model(n_sensors, n_chunks, config.vector_dims)

ckpt = torch.load(dataset_name + '/.pth')
model.load_state_dict(ckpt)
model.eval()

### Step 4: evaluation

test_id = int(sys.argv[2])
sensors = eval_data[test_id - config.n_train][0] \
    if test_id >= config.n_train else train_data[test_id][0]
sensors[[2]] = 0
pred_vecs = model(sensors).detach().numpy().reshape(-1, config.vector_dims)

gt_vecs = eval_data[test_id - config.n_train][1] \
    if test_id >= config.n_train else train_data[test_id][1]

### Step 5: assemble, visualize

assemble_start = time.time()
print('assemble start')

frame_clouds = {}
chunk_clouds = []

assemble_time = 0.0

for chunk_id, (vec, pred_vec, chunk) in enumerate(zip(vecs, pred_vecs, chunks)):
    # find closest distance (most matching frame) from database
    frame_id = np.argmin(np.linalg.norm(vec - pred_vec, axis=1))
    if chunk_id == 2:
        print(vec, pred_vec, gt_vecs[chunk_id * 2:chunk_id * 2 + 2], frame_id, vec[frame_id])

    # if frame_id not in frame_clouds:
    #   coords, colors = get_coords_and_colors(
    #     coord_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras],
    #     color_files[frame_id * config.n_cameras:(frame_id + 1) * config.n_cameras]
    #   )
    #   frame_clouds[frame_id] = get_cloud(coords, colors)

    start = time.time()

    chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (frame_id, chunk_id))['arr_0']
    chunk_cloud = open3d.geometry.PointCloud()
    chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
    chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])

    # chunk_cloud = frame_clouds[frame_id].crop(chunk)
    chunk_clouds.append(chunk_cloud)

    assemble_time += time.time() - start

assemble_end = time.time()
print(assemble_end - assemble_start, assemble_time)

### Step 6: remove ceiling

coords, colors = get_coords_and_colors(
    coord_files[test_id * config.n_cameras:(test_id + 1) * config.n_cameras],
    color_files[test_id * config.n_cameras:(test_id + 1) * config.n_cameras]
)
ground_truth = get_cloud(coords, colors)

ceiling = ground_truth.get_axis_aligned_bounding_box()

no_ceiling = open3d.geometry.AxisAlignedBoundingBox(
    np.array(ceiling.min_bound),
    np.array([ceiling.max_bound[0], ceiling.max_bound[1] - 1e-1, ceiling.max_bound[2]])
)  # any more elegant way to 'crop' a bounding box?

ground_truth = ground_truth.crop(no_ceiling)
chunk_clouds = [cloud.crop(no_ceiling) for cloud in chunk_clouds]

# open3d.visualization.draw_geometries([ground_truth])

chunk_clouds = merge_clouds(chunk_clouds)
open3d.visualization.draw_geometries([chunk_clouds])
