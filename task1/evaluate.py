# evaluate.py
import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

# 将 train 目录添加到 Python 搜索路径
# 这样可以方便地从 train.base_trainer 导入通用函数和默认配置
# 获取当前脚本 (evaluate.py) 所在的目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# train 目录的路径应该是 evaluate.py 所在目录下的 train 子目录
train_dir_path = os.path.join(current_script_dir, 'train')
if train_dir_path not in sys.path:
    sys.path.append(train_dir_path)

# 从 base_trainer 导入所需的函数和常量
# 确保 base_trainer.py 中定义了这些常量或可以从 config 传递
try:
    from train.base_trainer import get_data_loaders, evaluate_model, NUM_CLASSES, BATCH_SIZE, DEFAULT_DATA_PATH_TEST
except ImportError:
    print("错误：无法从 train.base_trainer 导入模块。请确保 evaluate.py 与 train/ 目录在同一级别或正确设置了PYTHONPATH。")
    # 定义一些备用常量，以防导入失败（但这通常表示项目结构或PYTHONPATH问题）
    NUM_CLASSES = 101
    BATCH_SIZE = 32
    DEFAULT_DATA_PATH_TEST = "/remote-home/yinzhitao/CVMidPJ-task1/data/test" # 请替换为您的实际默认路径
    # 如果导入失败，get_data_loaders 和 evaluate_model 将无法使用，脚本会出错。更好的做法是确保导入成功。
    print("警告：如果导入失败，脚本可能无法正常运行。将尝试使用内置的默认值。")


def main():
    parser = argparse.ArgumentParser(description="加载已训练的模型并进行评估。")

    # 必要参数
    parser.add_argument('--model_path', type=str, required=True, help="已训练模型的 .pth 文件路径。")
    parser.add_argument('--model_name', type=str, required=True, choices=['alexnet', 'resnet18', 'resnet34'], help="模型名称/架构 (例如: alexnet, resnet18, resnet34)。")
    

    # 数据和评估相关参数
    parser.add_argument('--data_path_test', type=str, default=DEFAULT_DATA_PATH_TEST, help=f"测试数据集路径 (默认: {DEFAULT_DATA_PATH_TEST})。")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f"评估时的批量大小 (默认: {BATCH_SIZE})。")
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES, help=f"模型输出的类别数量 (默认: {NUM_CLASSES})。")
    
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 根据 model_name (和可选的 model_arch) 实例化模型结构
    print(f"正在为评估实例化模型: {args.model_name}")
    if args.model_name == 'alexnet':
        model = models.alexnet(pretrained=False) # pretrained=False 因为我们要加载自己的权重
        # 修改分类器以匹配类别数 (与训练时一致)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_classes)
    elif args.model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model_name == 'resnet34':
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    # 如果支持更多模型，在此添加 elif 分支
    else:
        print(f"错误: 不支持的模型名称 '{args.model_name}'。请从 ['alexnet', 'resnet18', 'resnet34'] 中选择。")
        return

    model = model.to(device) # 模型移动到设备

    # 2. 加载已保存的模型权重
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件未找到于路径 '{args.model_path}'")
        return
    
    try:
        print(f"正在从 '{args.model_path}' 加载模型权重...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        print("请确保模型架构 (--model_name) 与 .pth 文件中的权重匹配。")
        return

    # 3. 加载测试数据
    try:
        _, test_loader = get_data_loaders(
            data_path_train=None, # 评估时不需要训练数据加载器
            data_path_test=args.data_path_test,
            batch_size=args.batch_size
        )
    except NameError: # 如果 base_trainer 中的函数没有成功导入
        print("错误: get_data_loaders 函数未定义。请检查 train.base_trainer 模块的导入。")
        return
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return
        
    if test_loader is None:
        print(f"错误:未能加载测试数据从路径 {args.data_path_test}")
        return

    criterion = nn.CrossEntropyLoss().to(device)

    # 5. 执行评估
    print("\n--- 开始评估 ---")
    try:
        test_loss, test_accuracy = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            phase="最终评估 (Final Evaluation)" 
        )
        print("--- 评估完成 ---")

    except NameError:
        print("错误: evaluate_model 函数未定义。请检查 train.base_trainer 模块的导入。")
        return
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        return

if __name__ == "__main__":
    main()