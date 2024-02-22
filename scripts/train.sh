#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
# PRETRAINED_MODEL_PATH=???   # path to pretrained avhubert
# DATA_PATH=???   # path to train dataset dir
# LLM_PATH=???    # path to llama checkpoint
# OUT_PATH=???    # output path to save 
PRETRAINED_MODEL_PATH=/mnt/usb/jh/Exp/acl24/av_hubert/avhubert/avhubert_checkpoints/large_vox_iter5.pt   # path to pretrained avhubert
DATA_PATH=/mnt/usb/jh/Exp/acl24/unit_qlora/v_based_multitask/433h/200   # path to train dataset dir
LLM_PATH=/mnt/usb/jh/Exp/acl24/av_hubert/avhubert/llm_checkpoints/llama/Llama-2-7b-hf    # path to llama checkpoint
OUT_PATH=/mnt/usb/jh/Exp/acl24/VSP-LLM/src/exp/test    # output path to save 

ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
SRC=${ROOT}/src

# start training
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
fairseq-hydra-train \
    --config-dir ./src/conf \
    --config-name avhubert_llama_cluster_trans_train \
        common.user_dir=${SRC} \
        task.data=${DATA_PATH} \
        task.label_dir=${DATA_PATH} \
        task.llm_ckpt_path=${LLM_PATH} \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.llm_ckpt_path=${LLM_PATH} \
        hydra.run.dir=${OUT_PATH} \
        distributed_training.distributed_world_size=1 \
        distributed_training.nprocs_per_node=1 \
        optimization.lr=[0.001] \
        optimization.update_freq=[4] \
        optimization.max_update=10000 \
        lr_scheduler.warmup_steps=5000 \
        lr_scheduler.decay_steps=10000