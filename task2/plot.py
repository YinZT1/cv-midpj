import re
import matplotlib.pyplot as plt

log_file = './mask-rcnn_r50_fpn_2x.log'
log_file = './mask-rcnn_r50_fpn_2x_pretrained.log'
# log_file = './sparse-rcnn_r50_fpn_2x_pretrained.log'

train_losses = []
train_loss_cls = []
train_loss_bbox = []
train_loss_mask = []
val_bbox_mAP = []
val_segm_mAP = []

train_loss_pattern = re.compile(
    r'Epoch\(train\) \[\d+\]\[\d+/\d+\] .* loss: ([\d.]+)')
train_loss_cls_pattern = re.compile(
    r'Epoch\(train\) \[\d+\]\[\d+/\d+\] .* loss_cls: ([\d.]+)')
train_loss_bbox_pattern = re.compile(
    r'Epoch\(train\) \[\d+\]\[\d+/\d+\] .* loss_bbox: ([\d.]+)')
train_loss_mask_pattern = re.compile(
    r'Epoch\(train\) \[\d+\]\[\d+/\d+\] .* loss_mask: ([\d.]+)')
val_bbox_mAP_pattern = re.compile(
    r'Epoch\(val\) \[\d+\]\[\d+/\d+\] .* coco/bbox_mAP: ([\d.]+)')
val_segm_mAP_pattern = re.compile(
    r'Epoch\(val\) \[\d+\]\[\d+/\d+\] .* coco/segm_mAP: ([\d.]+)')

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        train_loss_match = train_loss_pattern.search(line)
        train_loss_cls_match = train_loss_cls_pattern.search(line)
        train_loss_bbox_match = train_loss_bbox_pattern.search(line)
        train_loss_mask_match = train_loss_mask_pattern.search(line)
        val_bbox_mAP_match = val_bbox_mAP_pattern.search(line)
        val_segm_mAP_match = val_segm_mAP_pattern.search(line)
        if train_loss_match:
            train_losses.append(float(train_loss_match.group(1)))
        if train_loss_cls_match:
            train_loss_cls.append(float(train_loss_cls_match.group(1)))
        if train_loss_bbox_match:
            train_loss_bbox.append(float(train_loss_bbox_match.group(1)))
        if train_loss_mask_match:
            train_loss_mask.append(float(train_loss_mask_match.group(1)))
        if val_bbox_mAP_match:
            val_bbox_mAP.append(float(val_bbox_mAP_match.group(1)))
        if val_segm_mAP_match:
            val_segm_mAP.append(float(val_segm_mAP_match.group(1)))


# 绘制训练集上的总损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Total Loss', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Total Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# 绘制训练集上的其他损失曲线
plt.figure(figsize=(12, 6))
if train_loss_cls:
    plt.plot(train_loss_cls, label='Loss Cls', color='orange')
if train_loss_bbox:
    plt.plot(train_loss_bbox, label='Loss BBox', color='green')
if train_loss_mask:
    plt.plot(train_loss_mask, label='Loss Mask', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curves (Cls, BBox, Mask)')
plt.legend()
plt.grid(True)
plt.show()

# 绘制验证集上的 mAP 曲线
plt.figure(figsize=(12, 6))
plt.plot(val_bbox_mAP, label='Validation bbox_mAP', color='blue')
plt.plot(val_segm_mAP, label='Validation segm_mAP', color='green')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('Validation mAP Curves')
plt.legend()
plt.grid(True)
plt.show()
