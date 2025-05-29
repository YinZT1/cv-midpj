from mmdet.apis import DetInferencer

inferencer = DetInferencer(
    model='../configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py',
    weights='./sparse-rcnn_r50_fpn_2x_pretrained.pth'
)


# img_files = [
#     '000025.jpg',
#     '000031.jpg',
#     '000049.jpg',
#     '000058.jpg'
# ]
# img_files = [
#     'extra1.jpg',
#     'extra2.jpg',
#     'extra3.jpg'
# ]
img_files = [
    '11.jpg'
]


for path in img_files:
    # inferencer(path, show=False,out_dir='outputs/mask-rcnn_r50_fpn_2x_pretrained/')
    # inferencer(path, show=False,out_dir='outputs/mask-rcnn_r50_fpn_2x/')
    inferencer(path, show=False,
               out_dir='outputs/sparse-rcnn_r50_fpn_2x_pretrained/')
