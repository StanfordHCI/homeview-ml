# dataset generated from virtualhome, shortened prefix as vh
dataset_prefix = 'vh'

# number of cameras per frame
n_cameras = 20

# index of reference frame
ref_frame_id = 0

# length of chunk along x/z axis (in meters)
chunk_size = 1

# frames split to train set
n_train = 100

# dimensions of the chunk feature
feature_dims = 2

# learning rate
lr = 0.1

# minimum learning rate
min_lr = 0.000001

# training epochs
epochs = 4096

# minibatch size 
batch_size = 4

# number of epochs between evaluations
eval_freq = 4

# number of epochs between saves
save_freq = 64

# threshold to decide if a chunk should be updated (used in app.py)
epsilon = 2e-1