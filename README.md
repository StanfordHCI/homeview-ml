### Dataset Preparation

Step 1. Place the generated file in directory `vh.[name]/raw/`. Make sure the frame count exceeds `[n_train]` set in config.py. Each frame should contain one frame_id.json and `[n_cameras]` of frame_id-camera_id-point_cloud.exr and `[n_cameras]` of frame_id-camera_id-rgb.png.

Step 2. Convert to dataset. The first `[n_train]` frames are used for training and the rest for evaluation.

```bash
python vh.py [name]
```
train.pth and eval.pth will be generated and saved in `vh.[name]/`.


### Train

```bash
python train.py [name]
```

### Test

Specify the frame id `[eval_id]` for evaluation.

```bash
python test.py [name] [eval_id]
```

### Demo-Backend

#### 1. install requirements

```bash
pip install flask flask-compress
```

#### 2. prepare chunks

prepare locally

```bash
python localize.py [name]
```

or download the chunks [here](https://drive.google.com/file/d/1JvMEulAknZuQVcUlbhd0XsEzAod1uodK/view?usp=sharing), then extract it to `vh.[name]/chunks`.

#### 3. run backend

```bash
python app.py [name]
```