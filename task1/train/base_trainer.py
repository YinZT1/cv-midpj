# train/base_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
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
        data_path_train (str, optional): 训练数据路径。如果为None，则不加载训练数据。
        data_path_test (str, optional): 测试数据路径。如果为None，则不加载测试数据。
        batch_size (int): 批量大小。
    返回:
        tuple: (train_loader, test_loader)。如果对应路径为None，则加载器也为None。
    """
    train_loader = None
    test_loader = None

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
        if data_path_train: # 仅当提供了训练数据路径时才加载
            print(f"尝试从 {data_path_train} 加载训练数据...")
            train_dataset = datasets.ImageFolder(data_path_train, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            print(f"成功从 {data_path_train} 加载训练数据 ({len(train_dataset)} 个样本)")
            if len(train_dataset.classes) != NUM_CLASSES:
                 print(f"警告: 训练集类别数量 ({len(train_dataset.classes)}) 与预期 ({NUM_CLASSES}) 不匹配。")
        else:
            print("未提供训练数据路径，跳过加载训练数据。")

        if data_path_test: # 仅当提供了测试数据路径时才加载
            print(f"尝试从 {data_path_test} 加载测试数据...")
            test_dataset = datasets.ImageFolder(data_path_test, transform=test_transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            print(f"成功从 {data_path_test} 加载测试数据 ({len(test_dataset)} 个样本)")
            if len(test_dataset.classes) != NUM_CLASSES:
                 print(f"警告: 测试集类别数量 ({len(test_dataset.classes)}) 与预期 ({NUM_CLASSES}) 不匹配。")
        else:
            print("未提供测试数据路径，跳过加载测试数据。")
            
        return train_loader, test_loader

    except FileNotFoundError as e:
        # 根据哪个路径是 None 或无效来提供更具体的错误信息
        missing_path = ""
        if data_path_train and not os.path.exists(data_path_train): # 仅当尝试加载时检查
            missing_path = data_path_train
        elif data_path_test and not os.path.exists(data_path_test): # 仅当尝试加载时检查
             missing_path = data_path_test

        if missing_path:
            print(f"错误: 数据集路径 '{missing_path}' 未找到。请检查路径。")
        else:
            # 如果错误不是因为路径不存在，而是其他原因（例如路径是None但仍被错误使用）
            print(f"错误: 加载数据集时发生文件未找到错误。可能原因：路径配置问题。详细信息: {e}")
        exit(1)
    except Exception as e:
        print(f"加载数据集时发生未预期错误: {e}")
        exit(1)

# ... (base_trainer.py 中的其他函数 train_one_epoch, evaluate_model, plot_training_results, run_training_session 保持不变) ...
# (确保将 evaluate_model, plot_training_results, run_training_session 的代码也包含在 base_trainer.py 中)
# (这里为了简洁，只显示了 get_data_loaders 的修改)

# 重新粘贴其余的 base_trainer.py 函数，确保它们在同一个文件中
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, total_epochs, scheduler=None):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        if batch_idx % 50 == 0:
            print(f"  轮次 {epoch_num}/{total_epochs}, 批次 {batch_idx}/{len(loader)}, 训练损失: {loss.item():.4f}")
    if scheduler:
        scheduler.step()
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
    current_lr_group0 = optimizer.param_groups[0]['lr'] if optimizer.param_groups else "N/A"
    print(f"轮次 {epoch_num}/{total_epochs} 总结: 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_accuracy:.2f}%, 当前学习率(组0): {current_lr_group0:.6f}")
    return epoch_loss, epoch_accuracy

def evaluate_model(model, loader, criterion, device, epoch_num=None, total_epochs=None, phase="测试(Test)"):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    if loader is None: # 如果loader是None (例如测试数据路径未提供)
        print(f"警告: {phase} 数据加载器未定义，跳过评估阶段。")
        return 0.0, 0.0 # 返回默认值
    with torch.no_grad():
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
    train_losses = results_history['train_losses']
    train_accs = results_history['train_accuracies']
    test_losses = results_history['test_losses']
    test_accs = results_history['test_accuracies']
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='训练损失 (Train Loss)')
    plt.plot(epochs_range, test_losses, label='测试损失 (Test Loss)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.title(f'{plot_title_prefix} - 损失曲线', fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='训练准确率 (Train Accuracy)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (%)')
    plt.title(f'{plot_title_prefix} - 训练准确率曲线', fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, test_accs, label='测试准确率 (Test Accuracy)', color='red')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (%)')
    plt.title(f'{plot_title_prefix} - 测试准确率曲线', fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()
    print(f"训练过程图已保存至: {plot_save_path}")

def run_training_session(config, work_dir_base, figs_dir_base):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_tag_suffix = f"_{config.get('run_tag', '')}" if config.get('run_tag') else ""
    if config['train_mode'] == 'scratch':
        hyperparam_suffix = f"lr{config['lr']}_wd{config['weight_decay']}"
    elif config['train_mode'] == 'finetune':
        lr_cls_key = 'lr_classifier'
        hyperparam_suffix = f"lrcls{config[lr_cls_key]}_lrother{config['lr_other']}_wd{config['weight_decay']}"
    else:
        hyperparam_suffix = "unknown_params"
    base_filename = f"{config['model_name']}_{config.get('model_arch', '')}_{config['train_mode']}_{hyperparam_suffix}_ep{config['num_epochs']}{run_tag_suffix}_{run_timestamp}"
    model_save_path = os.path.join(work_dir_base, f"{base_filename}.pth")
    plot_save_path = os.path.join(figs_dir_base, f"{base_filename}_plot.png")
    plot_title_prefix = f"{config['model_name']} (架构: {config.get('model_arch', 'N/A')} - 模式: {config['train_mode']})"
    if config['train_mode'] == 'scratch':
        plot_title_prefix += f" lr={config['lr']} wd={config['weight_decay']}"
    else: 
        plot_title_prefix += f" lr_cls={config.get('lr_classifier','N/A')} lr_other={config.get('lr_other','N/A')} wd={config['weight_decay']}"
    if config.get('run_tag'):
        plot_title_prefix += f" 标签={config['run_tag']}"
    train_loader, test_loader = get_data_loaders(
        data_path_train=config.get('data_path_train', DEFAULT_DATA_PATH_TRAIN),
        data_path_test=config.get('data_path_test', DEFAULT_DATA_PATH_TEST),
        batch_size=config.get('batch_size', BATCH_SIZE)
    )
    if config['model_name'] == 'alexnet':
        from .alexnet_trainer import get_alexnet_model_optimizer 
        model, optimizer, scheduler = get_alexnet_model_optimizer(config, device, NUM_CLASSES)
    elif config['model_name'] == 'resnet18': 
        from .resnet_trainer import get_resnet_model_optimizer
        model, optimizer, scheduler = get_resnet_model_optimizer(config, device, NUM_CLASSES)
    else:
        raise ValueError(f"不支持的模型名称: {config['model_name']}")
    criterion = nn.CrossEntropyLoss().to(device)
    best_test_acc = 0.0
    results_history = {
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
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config['num_epochs'], scheduler)
        results_history['train_losses'].append(train_loss)
        results_history['train_accuracies'].append(train_acc)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device, epoch, config['num_epochs'])
        results_history['test_losses'].append(test_loss)
        results_history['test_accuracies'].append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  已保存最佳模型，测试准确率: {best_test_acc:.2f}%")
    print(f"--- 训练会话结束 ---")
    print(f"本次运行的最佳测试准确率: {best_test_acc:.2f}%")
    print(f"模型已保存至: {model_save_path}")
    plot_training_results(results_history, config['num_epochs'], plot_title_prefix, plot_save_path)
    run_summary = {
        'config': config,
        'best_test_accuracy': best_test_acc,
        'model_path': model_save_path,
        'plot_path': plot_save_path,
        'full_history': results_history
    }
    return run_summary