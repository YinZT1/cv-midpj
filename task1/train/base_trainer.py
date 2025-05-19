# train/base_trainer.py
import torch
import torch.nn as nn # 确保导入nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets # 使用 datasets 别名
import torchvision.models as models # 使用 models 别名
import os
import matplotlib.pyplot as plt
import time

# --- 全局常量 (可以被主脚本的配置覆盖) ---
NUM_CLASSES = 101 # 数据集类别数
BATCH_SIZE = 32   # 默认批量大小
DEFAULT_DATA_PATH_TRAIN = "/remote-home/yinzhitao/CVMidPJ-task1/data/train" # 默认训练数据路径
DEFAULT_DATA_PATH_TEST = "/remote-home/yinzhitao/CVMidPJ-task1/data/test"   # 默认测试数据路径


def get_data_loaders(data_path_train=DEFAULT_DATA_PATH_TRAIN,
                     data_path_test=DEFAULT_DATA_PATH_TEST,
                     batch_size=BATCH_SIZE):
    """
    加载并返回训练和测试数据加载器。
    参数:
        data_path_train (str): 训练数据路径。
        data_path_test (str): 测试数据路径。
        batch_size (int): 批量大小。
    返回:
        tuple: (train_loader, test_loader)
    """
    # 训练数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomCrop(224, padding=4), # 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 颜色抖动
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet均值和标准差归一化
    ])
    # 测试数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = datasets.ImageFolder(data_path_train, transform=train_transform)
        test_dataset = datasets.ImageFolder(data_path_test, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"成功从 {data_path_train} 加载训练数据 ({len(train_dataset)} 个样本)")
        print(f"成功从 {data_path_test} 加载测试数据 ({len(test_dataset)} 个样本)")
        if len(train_dataset.classes) != NUM_CLASSES or len(test_dataset.classes) != NUM_CLASSES:
            print(f"警告: 数据集类别数量不匹配。预期 {NUM_CLASSES}, 实际找到 {len(train_dataset.classes)}(训练集)/{len(test_dataset.classes)}(测试集)")
        return train_loader, test_loader
    except FileNotFoundError as e:
        print(f"错误: 数据集未找到。请检查路径: {data_path_train} 或 {data_path_test}")
        print(e)
        exit(1) # 出现错误时退出
    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        exit(1)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, total_epochs, scheduler=None):
    """
    执行一个训练周期 (epoch)。
    参数:
        model (torch.nn.Module): 待训练的模型。
        loader (DataLoader): 训练数据加载器。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (torch.device): 计算设备 (cpu 或 cuda)。
        epoch_num (int): 当前周期数。
        total_epochs (int): 总周期数。
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。
    返回:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device) # 数据移动到设备
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item() * images.size(0) # 累加批次损失乘以样本数
        _, predicted = torch.max(outputs, 1) # 获取预测类别
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item() # 计算正确预测数

        if batch_idx % 50 == 0: # 每50个批次打印一次进度
            print(f"  轮次 {epoch_num}/{total_epochs}, 批次 {batch_idx}/{len(loader)}, 训练损失: {loss.item():.4f}")
    
    if scheduler: # 如果使用了学习率调度器，则更新学习率
        scheduler.step()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
    # 获取当前优化器第一组参数的学习率（通常对于SGD就是整体学习率）
    current_lr_group0 = optimizer.param_groups[0]['lr'] if optimizer.param_groups else "N/A"
    print(f"轮次 {epoch_num}/{total_epochs} 总结: 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_accuracy:.2f}%, 当前学习率(组0): {current_lr_group0:.6f}")
    return epoch_loss, epoch_accuracy


def evaluate_model(model, loader, criterion, device, epoch_num=None, total_epochs=None, phase="测试(Test)"):
    """
    在给定的数据集上评估模型。
    参数:
        model (torch.nn.Module): 待评估的模型。
        loader (DataLoader): 数据加载器。
        criterion (torch.nn.Module): 损失函数。
        device (torch.device): 计算设备。
        epoch_num (int, optional): 当前周期数 (用于打印信息)。
        total_epochs (int, optional): 总周期数 (用于打印信息)。
        phase (str): 阶段名称，如 "测试(Test)" 或 "验证(Validation)"。
    返回:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 在评估阶段不计算梯度
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
    
    log_prefix = f"轮次 {epoch_num}/{total_epochs}, " if epoch_num and total_epochs else ""
    print(f"{log_prefix}{phase}损失: {epoch_loss:.4f}, {phase}准确率: {epoch_accuracy:.2f}%")
    return epoch_loss, epoch_accuracy


def plot_training_results(results_history, num_epochs, plot_title_prefix, plot_save_path):
    """
    绘制并保存训练过程中的损失和准确率曲线图。
    参数:
        results_history (dict): 包含 'train_losses', 'train_accuracies', 'test_losses', 'test_accuracies' 的字典。
        num_epochs (int): 总训练轮数。
        plot_title_prefix (str): 图像标题的前缀。
        plot_save_path (str): 图像保存的完整路径。
    """
    train_losses = results_history['train_losses']
    train_accs = results_history['train_accuracies']
    test_losses = results_history['test_losses']
    test_accs = results_history['test_accuracies']

    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 6)) # 设置图像大小

    # 子图1: 损失 vs. 轮次
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='训练损失 (Train Loss)')
    plt.plot(epochs_range, test_losses, label='测试损失 (Test Loss)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.title(f'{plot_title_prefix} - 损失曲线', fontsize=10)
    plt.legend()
    plt.grid(True)

    # 子图2: 训练准确率 vs. 轮次
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='训练准确率 (Train Accuracy)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (%)')
    plt.title(f'{plot_title_prefix} - 训练准确率曲线', fontsize=10)
    plt.legend()
    plt.grid(True)
    
    # 子图3: 测试准确率 vs. 轮次
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, test_accs, label='测试准确率 (Test Accuracy)', color='red')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (%)')
    plt.title(f'{plot_title_prefix} - 测试准确率曲线', fontsize=10)
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(plot_save_path)
    plt.close() # 关闭图像，防止在Jupyter等环境中重复显示
    print(f"训练过程图已保存至: {plot_save_path}")


def run_training_session(config, work_dir_base, figs_dir_base):
    """
    根据给定的配置运行单个完整的训练会话。
    参数:
        config (dict): 包含所有实验参数的配置字典。
        work_dir_base (str): 保存模型的根目录。
        figs_dir_base (str): 保存图像的根目录。
    返回:
        dict: 包含本次运行结果摘要的字典。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # 清理GPU缓存 (可选)

    run_timestamp = time.strftime("%Y%m%d-%H%M%S") # 当前时间戳，用于文件名
    run_tag_suffix = f"_{config.get('run_tag', '')}" if config.get('run_tag') else "" # 获取可选的运行标签

    # 构建详细的文件名后缀，包含关键超参数
    if config['train_mode'] == 'scratch':
        hyperparam_suffix = f"lr{config['lr']}_wd{config['weight_decay']}"
    elif config['train_mode'] == 'finetune':
        # 对于微调，lr_classifier可能对应AlexNet的分类器或ResNet的fc层
        lr_cls_key = 'lr_classifier' # 默认使用lr_classifier
        # 考虑到ResNet的配置可能使用lr_fc，这里可以增加灵活性，但当前脚本的main_trainer.py统一为lr_classifier
        hyperparam_suffix = f"lrcls{config[lr_cls_key]}_lrother{config['lr_other']}_wd{config['weight_decay']}"
    else:
        hyperparam_suffix = "unknown_params" # 未知参数的备用后缀
    
    # 基础文件名，包含模型、架构、模式、超参数、轮次、标签和时间戳
    base_filename = f"{config['model_name']}_{config.get('model_arch', '')}_{config['train_mode']}_{hyperparam_suffix}_ep{config['num_epochs']}{run_tag_suffix}_{run_timestamp}"

    model_save_path = os.path.join(work_dir_base, f"{base_filename}.pth") # 模型保存完整路径
    plot_save_path = os.path.join(figs_dir_base, f"{base_filename}_plot.png") # 图像保存完整路径
    
    # 构建图像标题前缀
    plot_title_prefix = f"{config['model_name']} (架构: {config.get('model_arch', 'N/A')} - 模式: {config['train_mode']})"
    if config['train_mode'] == 'scratch':
        plot_title_prefix += f" lr={config['lr']} wd={config['weight_decay']}"
    else: # finetune
        plot_title_prefix += f" lr_cls={config.get('lr_classifier','N/A')} lr_other={config.get('lr_other','N/A')} wd={config['weight_decay']}"
    if config.get('run_tag'):
        plot_title_prefix += f" 标签={config['run_tag']}"

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        data_path_train=config.get('data_path_train', DEFAULT_DATA_PATH_TRAIN),
        data_path_test=config.get('data_path_test', DEFAULT_DATA_PATH_TEST),
        batch_size=config.get('batch_size', BATCH_SIZE)
    )

    # 获取模型和优化器
    # 注意: 这里的导入使用了相对导入，因为base_trainer.py在train包内
    if config['model_name'] == 'alexnet':
        from .alexnet_trainer import get_alexnet_model_optimizer 
        model, optimizer, scheduler = get_alexnet_model_optimizer(config, device, NUM_CLASSES)
    elif config['model_name'] == 'resnet18': # 后续可扩展为检查 config['model_arch'] 以支持resnet34等
        from .resnet_trainer import get_resnet_model_optimizer
        model, optimizer, scheduler = get_resnet_model_optimizer(config, device, NUM_CLASSES)
    else:
        raise ValueError(f"不支持的模型名称: {config['model_name']}")
    
    criterion = nn.CrossEntropyLoss().to(device) # 损失函数

    # 训练循环的初始化
    best_test_acc = 0.0
    results_history = { # 用于存储每个epoch的结果
        'train_losses': [], 'train_accuracies': [],
        'test_losses': [], 'test_accuracies': []
    }

    print(f"\n--- 开始训练会话 ---")
    print(f"模型: {config['model_name']} (架构: {config.get('model_arch', 'N/A')}), 模式: {config['train_mode']}")
    print(f"轮数: {config['num_epochs']}, 批量大小: {config.get('batch_size', BATCH_SIZE)}")
    if config['train_mode'] == 'scratch':
        print(f"学习率: {config['lr']}, 权重衰减: {config['weight_decay']}")
    elif config['train_mode'] == 'finetune':
        print(f"学习率(分类器/FC): {config.get('lr_classifier','N/A')}, 学习率(其他层): {config.get('lr_other','N/A')}, 权重衰减: {config['weight_decay']}")
        print(f"预训练模型路径: {config.get('pretrained_path', 'N/A')}")
    if config.get('run_tag'):
        print(f"运行标签: {config['run_tag']}")
    if config.get('use_scheduler'):
        print(f"使用学习率调度器: StepLR (step={config.get('lr_step_size')}, gamma={config.get('lr_gamma')})")
    print(f"最佳模型将保存至: {model_save_path}")
    print(f"训练过程图将保存至: {plot_save_path}")
    print("-----------------------------")

    # 主训练循环
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config['num_epochs'], scheduler)
        results_history['train_losses'].append(train_loss)
        results_history['train_accuracies'].append(train_acc)

        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, epoch, config['num_epochs'])
        results_history['test_losses'].append(test_loss)
        results_history['test_accuracies'].append(test_acc)

        # 如果当前测试准确率更高，则保存模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  已保存最佳模型，测试准确率: {best_test_acc:.2f}%")
    
    print(f"--- 训练会话结束 ---")
    print(f"本次运行的最佳测试准确率: {best_test_acc:.2f}%")
    print(f"模型已保存至: {model_save_path}")

    # 绘制并保存训练结果图
    plot_training_results(results_history, config['num_epochs'], plot_title_prefix, plot_save_path)
    
    # 准备本次运行的摘要信息
    run_summary = {
        'config': config, # 保存实际使用的配置
        'best_test_accuracy': best_test_acc,
        'model_path': model_save_path,
        'plot_path': plot_save_path,
        'full_history': results_history # 可选：保存详细的epoch历史，用于更细致的分析
    }
    return run_summary