#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=???
MODEL_PATH=???  # path to trained model
DATA_ROOT=???   # path to test dataset dir
LLM_PATH=???   # path to llama checkpoint
OUT_PATH=???    # output path to save

# set paths
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
MODEL_SRC=${ROOT}/src

# fix variables based on langauge
if [[ $LANG == *"-"* ]] ; then
    TASK="avst"
    IFS='-' read -r SRC TGT <<< ${LANG}
    USE_BLEU=true
    if [[ $SRC == "en" ]] ; then
        DATA_PATH=${DATA_ROOT}/${TASK}/${SRC}/${TGT}
    else
        DATA_PATH=${DATA_ROOT}/${TASK}/${SRC}
    fi
else
    TASK="avsr"
    TGT=${LANG}
    USE_BLEU=false
    DATA_PATH=${DATA_ROOT}/${TASK}/${LANG}/${CLUSTER}
fi

# start decoding
export PYTHONPATH="${ROOT}/fairseq:$PYTHONPATH"
python -B ${MODEL_SRC}/vsp_llm_decode.py \
    --config-dir ${MODEL_SRC}/conf \
    --config-name s2s_decode \
        common.user_dir=${MODEL_SRC} \
        dataset.gen_subset=test \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        generation.beam=20 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.eval_bleu=${USE_BLEU} \
        override.llm_ckpt_path=${LLM_PATH} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH}/${TASK}/${LANG}