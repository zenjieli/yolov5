# Installation

**Pytorch** versions for CUDA 10.x is needed for training. If using **Pytorch** built for CUDA 11.x, one will get `nan` values for box losses. It is not needed to change the current 11.x CUDA installation.

```
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

# Training

```
python train.py --img 640 --batch 32 --epochs 300 --data person.yaml --weights pretrained.pt --cache
```

Option ```cache``` will load all images into RAM, which requires a lot of RAM (about 30 GB for 3 GB images).

# Detection

The source can be an image, video or a directory with images

```
python detect.py --weights best.pt --source /data/person/images/train --save-txt --nosave
```

`--save-txt`: Save detected bounding boxes in a text file for each image. The format is "id, left, top, width, height"
`--nosave`: Do not save images with detected bounding boxes

# Known issues

* GPU mode does not work for ONNX export

# Data preparation

## Annotation tool

## Annotation

<https://github.com/AlexeyAB/Yolo_mark>

```shell
cd ~/data/person

~/workspace/github/Yolo_mark/yolo_mark images/train train.txt names.txt
```