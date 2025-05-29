# 继承基础模型、数据集、训练计划和运行时设置
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',  # 我们会覆盖其中的大部分设置
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1. 模型修改：调整类别数量
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=20),  # VOC 数据集有 20 个类别
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=20)))  # VOC 数据集有 20 个类别

# 2. 数据集修改
dataset_type = 'CocoDataset'  # 数据集类型
data_root = 'data/VOCdevkit/coco/'  # 数据集根目录

# VOC 类别名称 (确保顺序与 JSON 文件中的 'categories' 一致)
# 这个 metainfo 非常重要，它会被传递给数据集和评估器
metainfo = {
    'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'),
    'palette': [
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
        (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0),
        (128, 192, 0), (0, 64, 128)
    ]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=[(1000, 600), (1200, 700)],  # 示例尺度范围，可调整
        keep_ratio=True,
        backend='pillow'  # 或者 'cv2'
    ),
    dict(type='PhotoMetricDistortion'),  # 添加颜色增强
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1000, 600),
         keep_ratio=True, backend='pillow'),  # 固定尺寸用于验证
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]


# 训练数据加载器
train_dataloader = dict(
    batch_size=10,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='voc0712_trainval.json',  # 训练集标注文件路径 (相对于 data_root)
        data_prefix=dict(img=''),  # 图片路径前缀 (相对于 data_root)
        pipeline=train_pipeline
    ))

# 验证数据加载器
val_dataloader = dict(
    batch_size=10,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='voc07_test.json',  # 验证集标注文件路径
        data_prefix=dict(img=''),
        pipeline=val_pipeline
    ))

# 测试数据加载器 (通常与验证数据加载器配置相同)
test_dataloader = val_dataloader
# test_dataloader = dict(dataset=dict(pipeline=val_pipeline))

# 3. 评估器修改
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc07_test.json',  # 与 val_dataloader 中的 ann_file 保持一致
    metric=['bbox', 'segm'],  # 同时评估边界框和实例分割
    format_only=False,
    # classwise=True # 可选，如果想查看每个类别的 AP 值
)
test_evaluator = val_evaluator

# 4. 训练计划修改 (可选)
# _base_/schedules/schedule_1x.py 默认是 1x 即 12 个 epoch
# 如果需要调整 epoch 或学习率策略，可以在这里覆盖
# 此处改为 2x 即 24 个 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))
param_scheduler = [  # 确保 param_scheduler 也被定义，如果 schedule_1x.py 中有的话
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],  # 2x schedule 的典型milestones
        gamma=0.1)
]

# 5. 加载预训练权重 (强烈推荐)
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
