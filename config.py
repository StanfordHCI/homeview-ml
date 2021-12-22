# number of cameras per frame
n_cameras = 20

# length of chunk along x/z axis
chunk_size = 1

# number of frames split to train set
n_train = 100

# learning rate
lr = 0.1

# minimum learning rate
min_lr = 0.0001

# training epochs
epochs = 4096

# minibatch size 
batch_size = 4

# number of epochs between evaluations
eval_freq = 4

# number of epochs between saves
save_freq = 64

# threshold to decide if a chunk should be updated
epsilon = 1e-2