#!/bin/bash

# ==============================================================================
# 示例 Shell 脚本，用于启动单个训练实验
# 请在使用前修改下面的 "配置区域" 中的参数
# ==============================================================================

# --- 配置区域 (根据您的实验需求修改这些值) ---
MODEL_NAME="resnet18"      # 可选: 'alexnet', 'resnet18'
MODEL_ARCH="resnet18"      # 具体架构: 'alexnet', 'resnet18', 'resnet34' 等 (如果model_name是resnet系列但此项未设置，通常默认为MODEL_NAME)
TRAIN_MODE="scratch"     # 可选: 'finetune' (微调), 'scratch' (从零训练)
NUM_EPOCHS=1             # 训练轮数 (示例值，请调大)
BATCH_SIZE=32            # 批量大小
WEIGHT_DECAY=0.0001      # 权重衰减

# --- 学习率配置 (根据 TRAIN_MODE 设置其中一组) ---
# 对于 'scratch' (从零训练) 模式:
LR_SCRATCH=0.01

# 对于 'finetune' (微调) 模式:
LR_CLASSIFIER=0.01   # 用于 AlexNet 的 classifier 层或 ResNet 的 fc 层
LR_OTHER=0.00001     # 用于微调模式下的其他层

# --- 路径配置 (如果 main_trainer.py 中的默认值不适用，请取消注释并修改) ---
# PRETRAINED_PATH_ARG="--pretrained_path /path/to/your/custom_resnet18.pth" # 微调模式下的预训练模型路径
# DATA_PATH_TRAIN_ARG="--data_path_train /custom/train_data"                # 自定义训练数据路径
# DATA_PATH_TEST_ARG="--data_path_test /custom/test_data"                  # 自定义测试数据路径
# WORK_DIR_ARG="--work_dir_base /custom/work_dir"                          # 自定义模型保存目录
# FIGS_DIR_ARG="--figs_dir_base /custom/figs_dir"                          # 自定义图像保存目录

# --- 学习率调度器配置 (可选) ---
USE_SCHEDULER_ARG="--use_scheduler" # 如果要使用调度器，请保留此行或设为 "--use_scheduler"
# USE_SCHEDULER_ARG=""                # 如果不使用调度器，请将此行注释掉或设为空字符串
LR_STEP_SIZE_ARG="--lr_step_size 20" # StepLR 的步长
LR_GAMMA_ARG="--lr_gamma 0.1"        # StepLR 的衰减因子

# --- 运行标签 (可选，用于更好地区分输出文件) ---
RUN_TAG="my_first_run" # 例如: "exp001_resnet_scratch_low_lr"

# ==============================================================================
# --- Python 可执行文件路径 ---
PYTHON_EXE="python" # 或者 "python3", 或者Python解释器的完整路径

# --- 构建传递给 main_trainer.py 的参数 ---
CMD_ARGS="--model_name ${MODEL_NAME}"
CMD_ARGS="${CMD_ARGS} --model_arch ${MODEL_ARCH}"
CMD_ARGS="${CMD_ARGS} --train_mode ${TRAIN_MODE}"
CMD_ARGS="${CMD_ARGS} --num_epochs ${NUM_EPOCHS}"
CMD_ARGS="${CMD_ARGS} --batch_size ${BATCH_SIZE}"
CMD_ARGS="${CMD_ARGS} --weight_decay ${WEIGHT_DECAY}"

# 根据训练模式添加相应的学习率参数
if [ "${TRAIN_MODE}" == "scratch" ]; then
  if [ -z "${LR_SCRATCH}" ]; then
    echo "错误: 'scratch' 模式下必须设置 LR_SCRATCH。" >&2
    exit 1
  fi
  CMD_ARGS="${CMD_ARGS} --lr ${LR_SCRATCH}"
elif [ "${TRAIN_MODE}" == "finetune" ]; then
  if [ -z "${LR_CLASSIFIER}" ] || [ -z "${LR_OTHER}" ]; then
    echo "错误: 'finetune' 模式下必须设置 LR_CLASSIFIER 和 LR_OTHER。" >&2
    exit 1
  fi
  CMD_ARGS="${CMD_ARGS} --lr_classifier ${LR_CLASSIFIER}"
  CMD_ARGS="${CMD_ARGS} --lr_other ${LR_OTHER}"
  # 如果为微调模式指定了预训练路径参数，则添加它
  if [ -n "${PRETRAINED_PATH_ARG}" ]; then
    CMD_ARGS="${CMD_ARGS} ${PRETRAINED_PATH_ARG}"
  fi
else
  echo "错误: 指定了无效的 TRAIN_MODE: ${TRAIN_MODE}" >&2
  exit 1
fi

# 添加自定义路径参数（如果已设置）
if [ -n "${DATA_PATH_TRAIN_ARG}" ]; then  CMD_ARGS="${CMD_ARGS} ${DATA_PATH_TRAIN_ARG}"; fi
if [ -n "${DATA_PATH_TEST_ARG}" ]; then   CMD_ARGS="${CMD_ARGS} ${DATA_PATH_TEST_ARG}"; fi
if [ -n "${WORK_DIR_ARG}" ]; then         CMD_ARGS="${CMD_ARGS} ${WORK_DIR_ARG}"; fi
if [ -n "${FIGS_DIR_ARG}" ]; then         CMD_ARGS="${CMD_ARGS} ${FIGS_DIR_ARG}"; fi

# 添加学习率调度器参数（如果启用）
if [ "${USE_SCHEDULER_ARG}" == "--use_scheduler" ]; then
  CMD_ARGS="${CMD_ARGS} ${USE_SCHEDULER_ARG}"
  CMD_ARGS="${CMD_ARGS} ${LR_STEP_SIZE_ARG}"
  CMD_ARGS="${CMD_ARGS} ${LR_GAMMA_ARG}"
fi

# 添加运行标签参数（如果已设置）
if [ -n "${RUN_TAG}" ]; then
  CMD_ARGS="${CMD_ARGS} --run_tag ${RUN_TAG}"
fi

# --- 执行训练脚本 ---
echo "即将执行命令:"
echo "${PYTHON_EXE} main_trainer.py ${CMD_ARGS}"
echo "--- 开始训练 ---"
${PYTHON_EXE} main_trainer.py ${CMD_ARGS}

# 获取退出状态码
EXIT_STATUS=$?
if [ ${EXIT_STATUS} -ne 0 ]; then
  echo "--- 训练失败，退出码: ${EXIT_STATUS} ---" >&2
else
  echo "--- 训练完成 ---"
fi

exit ${EXIT_STATUS}