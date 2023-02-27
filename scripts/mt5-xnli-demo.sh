#!/usr/bin/env bash

MODEL_DIR=${1:-${MODEL_DIR}}
TFDS_DATA_DIR=${2:-${TFDS_DATA_DIR}}

if [ -z ${MODEL_DIR} ] || [ -z ${TFDS_DATA_DIR} ]; then
  echo "usage: ./mt5-xnli-demo.sh gs://your-bucket/path/to/model_dir gs://your-bucket/path/to/tfds/cache"
  exit 1
fi

T5X_DIR="`python3 -m scripts.find_module t5x`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "============================="
PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/mt5_lm_adapted/base/checkpoint_1100000/checkpoint"

python3 -m t5x.train \
  --gin_search_paths="${T5X_DIR}" \
  --gin_file="t5x/examples/t5/mt5/base.gin" \
  --gin_file="t5x/configs/runs/finetune.gin" \
  --gin_file="configs/t5_finetune.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.MIXTURE_OR_TASK_NAME="'mt5_xnli_zeroshot'" \
  --gin.MIXTURE_OR_TASK_MODULE="'multilingual_t5.tasks'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 128}" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_120_000" \
  --gin.USE_CACHED_TASKS="False" \
  --tfds_data_dir=${TFDS_DATA_DIR}
