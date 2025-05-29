import os
import torch
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


config_file = '../configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc.py'
checkpoint_file = './mask-rcnn_r50_fpn_2x_pretrained.pth'
# checkpoint_file = './mask-rcnn_r50_fpn_2x.pth'
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

test_img_dir = './'
output_dir = './visualization/mask-rcnn_r50_fpn_2x_pretrained'
# output_dir = './visualization/mask-rcnn_r50_fpn_2x'
os.makedirs(output_dir, exist_ok=True)

img_files = [
    '000025.jpg',
    '000031.jpg',
    '000049.jpg',
    '000058.jpg'
]

# proposal 阶段只区分前景与背景
dataset_classes = ('front', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

# 初始化可视化器
visualizer = DetLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')]
)
visualizer.dataset_meta = {
    'classes': dataset_classes,
    'palette': [tuple(c) for c in model.dataset_meta['palette']]
}


def visualize_proposals_and_results(model, img_path, output_prefix, device):
    img = mmcv.imread(img_path)
    img_height, img_width = img.shape[:2]

    # 计算填充后的尺寸（pad_size_divisor=32）
    pad_height = ((img_height + 31) // 32) * 32
    pad_width = ((img_width + 31) // 32) * 32

    # 创建DetDataSample并设置正确的元信息
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape': (img_height, img_width, 3),
        'ori_shape': (img_height, img_width, 3),
        'pad_shape': (pad_height, pad_width, 3),
        'scale_factor': (1.0, 1.0),
        'img_norm_cfg': {
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True,
        }
    })

    # 准备模型输入（确保图像归一化与data_preprocessor一致）
    img_tensor = mmcv.imnormalize(
        img,
        mean=np.array([123.675, 116.28, 103.53]),
        std=np.array([58.395, 57.12, 57.375]),
        to_rgb=True,
    ).transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device).float()

    data = {
        'inputs': [img_tensor],
        'data_samples': [data_sample]
    }

    with torch.no_grad():
        x = model.extract_feat(data['inputs'][0])

        rpn_results_list = model.rpn_head.predict(
            x,
            data['data_samples']
        )

    # 提取proposals并转换为DetDataSample格式
    proposals = rpn_results_list[0].cpu()  # 第一张图像的proposals
    proposal_data_sample = DetDataSample()

    proposal_data_sample.pred_instances = InstanceData(
        bboxes=proposals.bboxes,
        scores=proposals.scores,
        labels=proposals.labels
    )

    # 可视化proposal boxes
    proposal_img = img.copy()
    visualizer.add_datasample(
        'proposals',
        proposal_img,
        data_sample=proposal_data_sample,  # 直接传递DetDataSample对象
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=f'{output_prefix}_proposals.jpg',
        pred_score_thr=0.05  # 低阈值以显示更多proposal
    )

    # 第二阶段：获取最终预测结果
    result = inference_detector(model, img_path)

    # 可视化最终预测结果（确保result是DetDataSample对象）
    if isinstance(result, list):
        result = result[0]  # inference_detector返回列表，取第一个元素
    result = result.cpu()

    visualizer.add_datasample(
        'predictions',
        img,
        data_sample=result,  # 直接传递DetDataSample对象
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=f'{output_prefix}_final.jpg',
        pred_score_thr=0.3  # 较高阈值过滤低置信度预测
    )

    print(
        f"可视化结果已保存至: {output_prefix}_proposals.jpg 和 {output_prefix}_final.jpg")


for img_file in img_files:
    img_path = os.path.join(test_img_dir, img_file)
    if os.path.exists(img_path):
        output_prefix = os.path.join(output_dir, os.path.splitext(img_file)[0])
        visualize_proposals_and_results(model, img_path, output_prefix, device)
    else:
        print(f"图像不存在: {img_path}")
