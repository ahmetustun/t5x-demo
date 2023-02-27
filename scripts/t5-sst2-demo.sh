#!/usr/bin/env bash

MODEL_DIR=${1:-${MODEL_DIR}}
TFDS_DATA_DIR=${2:-${TFDS_DATA_DIR}}

if [ -z ${MODEL_DIR} ] || [ -z ${TFDS_DATA_DIR} ]; then
  echo "usage: ./t5-sst2-demo.sh gs://your-bucket/path/to/model_dir gs://your-bucket/path/to/tfds/cache"
  exit 1
fi

T5X_DIR="`python3 -m scripts.find_module t5x`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "============================="
PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000"

python3 -m t5x.train \
  --gin_search_paths="${T5X_DIR}" \
  --gin_file="t5x/examples/t5/t5_1_1/base.gin" \
  --gin_file="t5x/configs/runs/finetune.gin" \
  --gin_file="configs/t5_finetune.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.MIXTURE_OR_TASK_NAME="'taskless_glue_sst2_v200_examples'" \
  --gin.MIXTURE_OR_TASK_MODULE="'prompt_tuning.data.glue'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 8}" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_120_000" \
  --gin.USE_CACHED_TASKS="False" \
  --tfds_data_dir=${TFDS_DATA_DIR}
