### Dataset Preparation

Step 1. Rename the Virtual-Home output directory to 'vh.[name]', for example: 'vh.door'
Make sure the directory contains 100+ frames of frame_id-camera_id-point_cloud.exr, frame_id.json, and frame_id-camera_id-rgb.png.

Step 2. Convert to dataset. The first 100 frames are used for training and the rest for evaluation.

```bash
python vh.py [name]
```
This generates vh.[name].train.pth and vh.[name].eval.pth.


### Train

```bash
python train.py [name]
```

### Test

Specify the frame id [eval_id] for evaluation.

```bash
python test.py [name] [eval_id]
```