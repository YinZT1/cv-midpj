# train/alexnet_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
def get_alexnet_model_optimizer(config, device, num_classes):
    """
    根据配置获取AlexNet模型和相应的优化器。
    参数:
        config (dict): 包含实验参数的配置字典。
        device (torch.device): 计算设备。
        num_classes (int): 输出类别的数量。
    返回:
        tuple: (model, optimizer, scheduler)
    """
    train_mode = config['train_mode'] # 训练模式: 'scratch' 或 'finetune'
    
    # 基础模型，预训练权重稍后根据模式处理
    model = models.alexnet(pretrained=False) 
    # 修改最后一个全连接层以适应数据集的类别数
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device) # 模型移动到设备

    optimizer = None
    scheduler = None

    if train_mode == 'finetune':
        # 微调模式
        pretrained_path = config.get('pretrained_path') # 获取预训练权重路径
        if pretrained_path and os.path.exists(pretrained_path): # 检查路径是否存在
            try:
                print(f"正在为 AlexNet 从以下路径加载预训练权重: {pretrained_path}")
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                model_dict = model.state_dict()
                # 过滤掉不匹配的键并确保形状一致
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False) # strict=False允许部分加载
                print("成功加载 AlexNet 的本地预训练权重。")
            except FileNotFoundError: # 虽然上面检查过，但以防万一
                print(f"警告: AlexNet 预训练权重在 {pretrained_path} 未找到。将对随机初始化的层进行微调。")
            except Exception as e:
                print(f"加载 AlexNet 预训练权重时出错: {e}。将使用随机初始化的权重。")
        else:
            if pretrained_path:
                 print(f"警告: AlexNet 预训练权重路径 '{pretrained_path}' 不存在或未提供。将对随机初始化的层进行微调。")
            else:
                 print(f"警告: 未给AlexNet微调模式提供 'pretrained_path'。将对随机初始化的层进行微调。")


        # 微调优化器 (分类器层和其他层使用不同学习率)
        lr_classifier = config['lr_classifier']
        lr_other = config['lr_other']
        weight_decay = config['weight_decay']
        
        classifier_params = list(model.classifier[6].parameters()) # 获取分类器最后一层的参数
        # 获取特征提取部分中需要梯度的参数
        other_params = [p for name, p in model.named_parameters() if 'classifier.6' not in name and p.requires_grad]
        
        optimizer = optim.SGD([
            {'params': classifier_params, 'lr': lr_classifier},
            {'params': other_params, 'lr': lr_other}
        ], momentum=0.9, weight_decay=weight_decay)
        print(f"优化器: SGD, 模式: 微调, 学习率(分类器): {lr_classifier}, 学习率(其他层): {lr_other}, 权重衰减: {weight_decay}")

    elif train_mode == 'scratch':
        # 从零开始训练模式 (所有参数使用相同的学习率)
        lr = config['lr']
        weight_decay = config['weight_decay']
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        print(f"优化器: SGD, 模式: 从零训练, 学习率: {lr}, 权重衰减: {weight_decay}")
    else:
        raise ValueError(f"AlexNet 不支持的训练模式: {train_mode}")

    # 根据配置决定是否使用学习率调度器
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=config.get('lr_step_size', 20), 
                                              gamma=config.get('lr_gamma', 0.1))
        print(f"使用 StepLR 学习率调度器: 步长={config.get('lr_step_size', 20)}, gamma因子={config.get('lr_gamma', 0.1)}")

    return model, optimizer, scheduler