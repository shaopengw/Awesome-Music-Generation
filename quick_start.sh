#!/bin/bash

export PYTHONPATH=/mnt/sda/upload_github/Awesome-Music-Generation:$PYTHONPATH

CONFIG_YAML="MMGen_train/config/quick_start/quick_start.yaml"
LIST_INFERENCE="tests/captionlist/test.lst"
RELOAD_FROM_CKPT="data/checkpoints/8wmusicbench+8kmusicaps-checkpoint-fad-133.00-global_step=11299.ckpt"

python3 MMGen_train/infer.py \
    --config_yaml $CONFIG_YAML \
    --list_inference $LIST_INFERENCE \
    --reload_from_ckpt $RELOAD_FROM_CKPT