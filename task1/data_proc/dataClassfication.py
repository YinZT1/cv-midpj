import os
import shutil
import random
from pathlib import Path
'''
该py文件对caltech-101进行分类。
'''

# 设置路径
data_root = "/root/YZT/CVMidPJ/data/caltech-101/101_ObjectCategories"  # 解压后的数据集根目录
train_dir = "/root/YZT/CVMidPJ/data/train"      # 训练集目标目录
test_dir = "/root/YZT/CVMidPJ/data/test"        # 测试集目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 按标准划分：每类30张训练，剩余最多50张测试
train_per_class = 30
test_max_per_class = 50

# 遍历每个类别文件夹
for category in os.listdir(data_root):
    category_path = os.path.join(data_root, category)
    if not os.path.isdir(category_path):
        continue
    
    # 获取所有图像文件
    images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
    random.shuffle(images)  # 随机打乱
    
    # 划分训练和测试
    train_images = images[:train_per_class]
    test_images = images[train_per_class:train_per_class + test_max_per_class]
    
    # 创建类别子目录
    train_category_dir = os.path.join(train_dir, category)
    test_category_dir = os.path.join(test_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)
    
    # 复制文件到对应目录
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_category_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_category_dir, img))

print("数据集划分完成！")