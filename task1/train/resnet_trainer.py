# train/resnet_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os # 用于 os.path.exists

def get_resnet_model_optimizer(config, device, num_classes):
    """
    根据配置获取ResNet模型 (如ResNet-18, ResNet-34) 和相应的优化器。
    参数:
        config (dict): 包含实验参数的配置字典。
        device (torch.device): 计算设备。
        num_classes (int): 输出类别的数量。
    返回:
        tuple: (model, optimizer, scheduler)
    """
    # model_arch 用于指定具体的ResNet架构，如'resnet18', 'resnet34'
    model_arch = config.get('model_arch', 'resnet18') # 默认为 'resnet18'
    train_mode = config['train_mode'] # 训练模式: 'scratch' 或 'finetune'

    # 根据 model_arch 构建模型实例
    if model_arch == 'resnet18':
        model = models.resnet18(pretrained=False) # 始终先设为False，权重加载在后面处理
    elif model_arch == 'resnet34':
        model = models.resnet34(pretrained=False)
    # 如果需要，可以在此添加更多ResNet变体 (e.g., resnet50, resnet101)
    else:
        raise ValueError(f"不支持的 ResNet 架构: {model_arch}")

    # 修改最后的全连接层 (fc) 以适应数据集的类别数
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device) # 模型移动到设备

    optimizer = None
    scheduler = None

    if train_mode == 'finetune':
        # 微调模式
        pretrained_path = config.get('pretrained_path') # 获取预训练权重路径
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                print(f"正在为 {model_arch} 从以下路径加载预训练权重: {pretrained_path}")
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                model_dict = model.state_dict()
                # 过滤掉不匹配的键 (特别是fc层，因为我们通常会重新训练它) 并确保形状一致
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape and 'fc.' not in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False) # strict=False 允许fc层不匹配或部分加载
                print(f"成功加载 {model_arch} 的本地预训练权重 (不包括fc层)。")
            except FileNotFoundError:
                print(f"警告: {model_arch} 预训练权重在 {pretrained_path} 未找到。将对随机初始化的层进行微调。")
            except Exception as e:
                print(f"加载 {model_arch} 预训练权重时出错: {e}。将使用随机初始化的权重。")
        else:
            if pretrained_path:
                print(f"警告: {model_arch} 预训练权重路径 '{pretrained_path}' 不存在或未提供。将对随机初始化的层进行微调。")
            else:
                print(f"警告: 未给 {model_arch} 微调模式提供 'pretrained_path'。将对随机初始化的层进行微调。")


        # 微调优化器 (fc层和其他层使用不同学习率)
        # 在main_trainer.py中，我们用lr_classifier来指代fc层的学习率，以保持与AlexNet配置项的统一
        lr_fc = config['lr_classifier'] 
        lr_other = config['lr_other']
        weight_decay = config['weight_decay']
        
        fc_params = list(model.fc.parameters()) # 获取fc层的参数
        # 获取除fc层以外其他所有需要梯度的参数
        other_params = [p for name, p in model.named_parameters() if 'fc' not in name and p.requires_grad]
        
        optimizer = optim.SGD([
            {'params': fc_params, 'lr': lr_fc},
            {'params': other_params, 'lr': lr_other}
        ], momentum=0.9, weight_decay=weight_decay)
        print(f"优化器: SGD, 模式: 微调, 学习率(FC层): {lr_fc}, 学习率(其他层): {lr_other}, 权重衰减: {weight_decay}")

    elif train_mode == 'scratch':
        # 从零开始训练模式 (所有参数使用相同的学习率)
        lr = config['lr']
        weight_decay = config['weight_decay']
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        print(f"优化器: SGD, 模式: 从零训练, 学习率: {lr}, 权重衰减: {weight_decay}")
    else:
        raise ValueError(f"ResNet 不支持的训练模式: {train_mode}")
    
    # 根据配置决定是否使用学习率调度器
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=config.get('lr_step_size', 20), 
                                              gamma=config.get('lr_gamma', 0.1))
        print(f"使用 StepLR 学习率调度器: 步长={config.get('lr_step_size', 20)}, gamma因子={config.get('lr_gamma', 0.1)}")

    return model, optimizer, scheduler