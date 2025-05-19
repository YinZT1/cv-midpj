# 图像分类项目 (Caltech101)

本项目提供了一个在 Caltech101 数据集上训练和评估图像分类模型（目前支持 AlexNet, ResNet-18/34）的框架。它支持从零开始训练模型，也支持微调预训练模型，所有配置均通过命令行参数传递。

## 项目结构

```

project\_root/ (项目根目录)
├── main\_trainer.py            \# 主Python脚本，接收命令行参数以运行单次实验
├── run\_single\_experiment.sh   \# 用于启动 main\_trainer.py 的示例Shell脚本
├── train/                     \# 包含模块化训练逻辑的目录
│   ├── **init**.py            \# 将 train 目录标记为Python包
│   ├── base\_trainer.py      \# 定义通用的训练、评估、绘图函数
│   ├── alexnet\_trainer.py   \# AlexNet特定的模型和优化器设置逻辑
│   └── resnet\_trainer.py    \# ResNet特定的模型和优化器设置逻辑
├── work\_dir/                  \# 保存训练好的最佳模型 (.pth文件) 的目录
├── figs/                      \# 保存训练过程图 (.png文件) 的目录
└── README.md                  \# 本项目说明文档
```

## 数据集准备

本项目默认配置为使用 Caltech101 数据集。

1.  下载并整理 Caltech101 数据集。

2.  `main_trainer.py` 脚本中定义了默认的数据集路径。您可以通过命令行参数覆盖这些默认值：

      * `--data_path_train /您存放训练数据的路径`
      * `--data_path_test /您存放测试数据的路径`

    期望的数据集目录结构应与 `torchvision.datasets.ImageFolder` 兼容（即，每个子目录以类别命名，包含该类别的所有图像）。

## 预训练模型 (微调模式可选)

如果您计划使用微调模式 (`--train_mode finetune`)：

1.  下载所需的预训练模型权重（例如，在 ImageNet 上预训练的 AlexNet 或 ResNet-18 权重）。
2.  `main_trainer.py` 脚本中定义了默认的预训练模型路径。您可以通过命令行参数 `--pretrained_path /您的预训练权重.pth` 来指定自定义路径。

如果在微调模式下未有效提供 `pretrained_path`，模型将使用随机初始化的权重进行训练，但如果指定了微调的学习率方案（即为分类器/FC层和其他层分别指定学习率），该方案仍然会生效。

## 运行实验

实验通过 `run_single_experiment.sh` 脚本启动，该脚本会调用 `main_trainer.py` 并传递相应的命令行参数。

**1. 配置 `run_single_experiment.sh`:**
打开 `run_single_experiment.sh` 文件，修改文件顶部的 "配置区域" 中的变量来定义您想运行的实验。

**2. 赋予脚本执行权限:**

```bash
chmod +x run_single_experiment.sh
```

**3. 运行实验:**

```bash
./run_single_experiment.sh
```

您也可以直接从命令行调用 `main_trainer.py` 并传递所有参数：

```bash
python main_trainer.py \
    --model_name resnet18 \
    --model_arch resnet18 \
    --train_mode scratch \
    --num_epochs 50 \
    --batch_size 32 \
    --lr 0.01 \
    --weight_decay 0.0001 \
    --use_scheduler \
    --lr_step_size 15 \
    --lr_gamma 0.1 \
    --run_tag my_resnet_scratch_run
```

### `main_trainer.py` 命令行参数详解:

  * `--model_name` (必需, 可选值: 'alexnet', 'resnet18'): 要训练的模型。
  * `--model_arch` (可选, 默认: 与 `model_name` 相同): 具体的模型架构 (例如: 'resnet18', 'resnet34')。
  * `--train_mode` (必需, 可选值: 'finetune', 'scratch'): 训练模式。
  * `--num_epochs` (必需, 整数): 训练的总轮数。
  * `--batch_size` (整数, 默认: 32): 批量大小。
  * `--weight_decay` (必需, 浮点数): 优化器的权重衰减系数。
  * `--lr` (浮点数): 学习率 (在 'scratch' 模式下必需)。
  * `--lr_classifier` (浮点数): 分类器/全连接层的学习率 (在 'finetune' 模式下必需)。
  * `--lr_other` (浮点数): 其他层的学习率 (在 'finetune' 模式下必需)。
  * `--pretrained_path` (字符串, 默认: None, 会使用脚本内为已知模型定义的默认路径): 'finetune' 模式下的预训练权重路径。
  * `--data_path_train` (字符串, 默认: 见脚本内定义): 训练数据路径。
  * `--data_path_test` (字符串, 默认: 见脚本内定义): 测试数据路径。
  * `--work_dir_base` (字符串, 默认: `./work_dir`): 保存模型的基础目录。
  * `--figs_dir_base` (字符串, 默认: `./figs`): 保存图像的基础目录。
  * `--use_scheduler` (标志, 默认: False): 是否使用 StepLR 学习率调度器。
  * `--lr_step_size` (整数, 默认: 20): StepLR 调度器的步长。
  * `--lr_gamma` (浮点数, 默认: 0.1): StepLR 调度器的 gamma 因子。
  * `--run_tag` (字符串, 默认: ""): 附加到输出文件名的可选标签，用于更好地区分不同运行的输出。

## 输出结果

  * **模型**: 每次运行中表现最佳（基于测试集准确率）的模型将以 `.pth` 格式保存在由 `--work_dir_base` 指定的目录中 (默认为 `project_root/work_dir/`)。
  * **图像**: 每次运行的训练/测试损失和准确率曲线图将以 `.png` 格式保存在由 `--figs_dir_base` 指定的目录中 (默认为 `project_root/figs/`)。

输出文件名将包含模型名称、具体架构、训练模式、关键超参数、轮数、可选的运行标签以及时间戳，以确保唯一性。
脚本运行结束后，会在终端打印本次运行的简要总结，包括最佳测试准确率以及模型和图像的保存路径。

## 自定义与扩展

  * **添加新模型**:
    1.  在 `train/` 目录下创建一个新的 `your_model_trainer.py` 文件。
    2.  在该文件中实现一个 `get_your_model_optimizer(config, device, num_classes)` 函数，其功能类似于 `alexnet_trainer.py` 或 `resnet_trainer.py` 中的同名函数。
    3.  修改 `train/base_trainer.py` 中的 `run_training_session` 函数（以及 `main_trainer.py` 中的 `model_name` choices），使其能够识别并调用您为新模型编写的训练器。
  * **更改优化器/学习率调度器**: 修改相应模型训练器文件 (`alexnet_trainer.py`, `resnet_trainer.py` 或您自定义的训练器) 中的优化器和调度器定义部分。
  * **数据增强**: 调整 `train/base_trainer.py` 文件中 `get_data_loaders` 函数内的 `train_transform` 定义。

