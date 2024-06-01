# How to use it to train and test

## Environment

python 3.12.3

pytorch 2.3.0

torchvision 0.18.0

tensorboard 2.14.0

tensorboardX 2.2

mmcv 2.1

mmdet 3.3

mmengine 0.10.4



## Data

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

## Data Switch

```
python voc_to_coco.py
python split.py
```

Switch data format from VOC to COCO

# Train

```
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_me.py  --work-dir faster_rcnn
```

# Test

```
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_me.py  --work-dir faster_rcnn
```
