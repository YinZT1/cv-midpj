# 继承基础模型、数据集、训练计划和运行时设置
_base_ = [
    '../_base_/datasets/coco_instance.py',  # 我们会覆盖其中的大部分设置
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1. 模型修改：调整类别数量
num_stages = 6
num_proposals = 100
model = dict(
    type='SparseRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=20,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# 2. 数据集修改
dataset_type = 'CocoDataset'  # 数据集类型
data_root = '../autodl-fs/data/VOCdevkit/coco/'  # 数据集根目录

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
    batch_size=8,
    num_workers=4,
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
# test_dataloader = val_dataloader
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))

# 3. 评估器修改
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc07_test.json',  # 与 val_dataloader 中的 ann_file 保持一致
    metric=['bbox'],  # 同时评估边界框和实例分割
    format_only=False,
    # classwise=True # 可选，如果想查看每个类别的 AP 值
)
test_evaluator = val_evaluator

# 4. 训练计划修改 (可选)
# _base_/schedules/schedule_1x.py 默认是 1x 即 12 个 epoch
# 如果需要调整 epoch 或学习率策略，可以在这里覆盖
# 此处改为 2x 即 24 个 epoch
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

# optimizer
# 在配置中添加混合精度训练支持
optim_wrapper = dict(
    _delete_=True,  # 替换原始 OptimWrapper， 务必 delete
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)
        }
    ),
    loss_scale='dynamic',  # 动态调整损失缩放因子
    dtype='float16'  # 显式指定FP16精度
)

param_scheduler = [  # 确保 param_scheduler 也被定义，如果 schedule_1x.py 中有的话
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],  # 2x schedule 的典型milestones
        # milestones=[12, 16, 20],
        gamma=0.1)
]

# 5. 加载预训练权重 (强烈推荐)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth'
