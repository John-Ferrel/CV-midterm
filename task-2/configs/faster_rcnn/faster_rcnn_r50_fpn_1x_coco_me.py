
# The new config inherits a base config to highlight the necessary modification

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)  # VOC has 20 classes
    )
)

# Modify dataset related settings
dataset_type = 'CocoDataset'
data_root = 'data/COCO/VOC2007/'
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow', 
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU used during training
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        type=dataset_type,
        ann_file='D:\CV Project\PJ2-2-me\data\COCO\VOC2007\annotations\train2007.json',
        img_prefix='D:\CV Project\PJ2-2-me\data\COCO\VOC2007\JPEGImages\train2024\\',
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        ann_file='D:\CV Project\PJ2-2-me\data\COCO\VOC2007\annotations\val2007.json',
        img_prefix='D:\CV Project\PJ2-2-me\data\COCO\VOC2007\JPEGImages\val2024\\',
        classes=classes
    ),
    test=dict(
        type=dataset_type,
        ann_file='D:\CV Project\PJ2-2-me\data\COCO\VOC2007\annotations\test2007.json',
        img_prefix='D:\CV Project\PJ2-2-me\data\COCO\VOC2007\JPEGImages\test2024\\',
        classes=classes
    )
)

# Modify schedule related settings
optim_wrapper = dict(  # 优化器封装的配置
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # 随机梯度下降优化器
        lr=0.02,  # 基础学习率
        # momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )
param_scheduler = [
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=10,   # 从第一个 epoch 开始
        end=30,  # 到第 12 个 epoch 结束
        milestones=[10, 20],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=30,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

# Modify runtime settings
checkpoint_config = dict(interval=1)

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50))
# 可选： 配置日志打印数值的平滑窗口大小
log_processor = dict(
    type='LogProcessor',
    window_size=50)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),

]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')