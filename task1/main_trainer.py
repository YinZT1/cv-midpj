# main_trainer.py
import argparse
import os
import sys
import torch # 确保导入torch以便进行设备检查

# 将 train 目录添加到 Python 搜索路径，以便直接导入其中的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

from train.base_trainer import run_training_session # 从base_trainer导入核心函数

# --- 基础路径定义 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # 项目根目录
DEFAULT_WORK_DIR_BASE = os.path.join(PROJECT_ROOT, "work_dir") # 默认模型保存路径
DEFAULT_FIGS_DIR_BASE = os.path.join(PROJECT_ROOT, "figs")     # 默认图像保存路径

# --- 默认数据集和预训练模型路径 (请根据您的实际情况更新) ---
DEFAULT_DATA_PATH_TRAIN = "/remote-home/yinzhitao/CVMidPJ-task1/data/train"
DEFAULT_DATA_PATH_TEST = "/remote-home/yinzhitao/CVMidPJ-task1/data/test"
DEFAULT_ALEXNET_PRETRAINED_PATH = "/remote-home/yinzhitao/CVMidPJ-task1/models/alexnet_pretrained.pth"
DEFAULT_RESNET18_PRETRAINED_PATH = "/remote-home/yinzhitao/CVMidPJ-task1/models/resnet18_pretrained.pth"

def main():
    parser = argparse.ArgumentParser(description="运行单次训练实验。")

    # 模型和训练模式相关参数
    parser.add_argument('--model_name', type=str, required=True, choices=['alexnet', 'resnet18'], help="要训练的模型 (可选: alexnet, resnet18)。")
    parser.add_argument('--model_arch', type=str, default=None, help="具体的模型架构，例如 'resnet18', 'resnet34'。如果model_name是resnet系列但此项未设置，则默认为model_name的值。")
    parser.add_argument('--train_mode', type=str, required=True, choices=['finetune', 'scratch'], help="训练模式: 'finetune' (微调) 或 'scratch' (从零开始训练)。")

    #核心超参数
    parser.add_argument('--num_epochs', type=int, required=True, help="训练的总轮数。")
    parser.add_argument('--batch_size', type=int, default=32, help="批量大小 (默认: 32)。")
    parser.add_argument('--weight_decay', type=float, required=True, help="优化器的权重衰减系数。")

    # 学习率 (根据训练模式条件性需要)
    parser.add_argument('--lr', type=float, help="学习率 (用于 'scratch' 模式)。")
    parser.add_argument('--lr_classifier', type=float, help="分类器/全连接层的学习率 (用于 'finetune' 模式)。")
    parser.add_argument('--lr_other', type=float, help="其他层的学习率 (用于 'finetune' 模式)。")

    # 路径相关参数
    parser.add_argument('--pretrained_path', type=str, default=None, help="预训练权重路径 (用于 'finetune' 模式)。如果未设置，将使用对应model_name的默认路径。")
    parser.add_argument('--data_path_train', type=str, default=DEFAULT_DATA_PATH_TRAIN, help=f"训练数据路径 (默认: {DEFAULT_DATA_PATH_TRAIN})。")
    parser.add_argument('--data_path_test', type=str, default=DEFAULT_DATA_PATH_TEST, help=f"测试数据路径 (默认: {DEFAULT_DATA_PATH_TEST})。")
    parser.add_argument('--work_dir_base', type=str, default=DEFAULT_WORK_DIR_BASE, help=f"保存模型的基础目录 (默认: {DEFAULT_WORK_DIR_BASE})。")
    parser.add_argument('--figs_dir_base', type=str, default=DEFAULT_FIGS_DIR_BASE, help=f"保存图像的基础目录 (默认: {DEFAULT_FIGS_DIR_BASE})。")

    # 学习率调度器 (可选)
    parser.add_argument('--use_scheduler', action='store_true', help="是否使用StepLR学习率调度器。")
    parser.add_argument('--lr_step_size', type=int, default=20, help="StepLR调度器的步长 (默认: 20)。")
    parser.add_argument('--lr_gamma', type=float, default=0.1, help="StepLR调度器的gamma因子 (默认: 0.1)。")
    
    # 运行标签 (可选，用于更好地区分输出文件)
    parser.add_argument('--run_tag', type=str, default="", help="附加到输出文件名的可选标签。")

    args = parser.parse_args()

    # 校验条件性学习率参数
    if args.train_mode == 'scratch' and args.lr is None:
        parser.error("'scratch' 训练模式下必须提供 --lr 参数。")
    if args.train_mode == 'finetune' and (args.lr_classifier is None or args.lr_other is None):
        parser.error("'finetune' 训练模式下必须提供 --lr_classifier 和 --lr_other 参数。")

    # 如果是微调模式且未提供预训练路径，则设置默认路径
    if args.train_mode == 'finetune' and args.pretrained_path is None:
        if args.model_name == 'alexnet':
            args.pretrained_path = DEFAULT_ALEXNET_PRETRAINED_PATH
        elif args.model_name == 'resnet18':
            args.pretrained_path = DEFAULT_RESNET18_PRETRAINED_PATH
        # 如果支持其他模型的微调，可以在此添加更多默认路径

    # 构建配置字典
    config = vars(args) # 将 argparse.Namespace 对象转换为字典

    # 确保 model_arch 被设置, 如果未指定，默认为 model_name (特别是resnet系列)
    if config.get('model_arch') is None:
        config['model_arch'] = config['model_name']

    # 确保输出目录存在
    os.makedirs(args.work_dir_base, exist_ok=True)
    os.makedirs(args.figs_dir_base, exist_ok=True)

    print("--- 实验配置 ---")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-------------------")

    # 调用核心训练函数
    summary = run_training_session(config, args.work_dir_base, args.figs_dir_base)

    print("\n--- 单次实验总结 ---")
    print(f"  模型: {config['model_name']} (模式: {config['train_mode']})")
    print(f"  最佳测试准确率: {summary['best_test_accuracy']:.2f}%")
    print(f"  模型保存路径: {summary['model_path']}")
    print(f"  图像保存路径: {summary['plot_path']}")
    print("-------------------")

if __name__ == "__main__":
    main()