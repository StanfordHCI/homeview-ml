Baseline method for Stanford HCI project 'Augmented Home Assistant'.

This branch only supports dataset generated from the VirtualHome simualtor.

### Dataset Preprocessing

1. Place the generated file in directory `vh.[dataset_name]/raw/`. Make sure the number of frames exceeds `[n_train]` as described in `config.py`. Each frame consists of exactly one `frame_id.json` and `[n_cameras]` of exr and png images respectively, formatted as `frame_id-camera_id-point_cloud.exr` and `frame_id-camera_id-rgb.png`.

2. Convert the image and json files to pytorch dataset. The first `[n_train]` frames are used for training and the rest for evaluation.

   ```
   python preprocess.py [dataset_name]
   ```

   `train.pth` and `eval.pth` will be generated and saved in `vh.[dataset_name]/`.


### Train

```
python train.py [dataset_name]
```

During and after the training process, `model.pth` will be saved as checkpoint.

### Test

Specify the frame id `[eval_id]` for evaluation.

```
python test.py [dataset_name] [eval_id]
```

### Backend

1. prepare chunks

   To avoid frequently decoding images on-the-fly, we precompute chunks of all frames and store it in `vh.[dataset_name]/chunk`, in the compressed npz format. Simply run

   ```
   python localize.py [dataset_name]
   ```
   
   100 frames will take 3GB of space approximately.
   
2. run backend

   ```
   python app.py [dataset_name]
   ```

### Other implemented functionalities

1. locating IoT sensors

   ```
   python locate.py [dataset_name]
   ```

2. objective performance comparison

   ```
   python stat.py [dataset_name]
   ```

