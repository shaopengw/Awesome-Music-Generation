#!/bin/bash
# Detailed Description of Key Parameters:

# Model Configuration Parameters
# --tmodel: Text encoder backbone (default: roberta)
# --amodel: Audio encoder backbone (HTSAT-tiny or HTSAT-base)
# --pretrained-audio: Path to pretrained audio encoder checkpoint

# Audio Encoder Options:
# 1. HTSAT-tiny:
#    Download: https://drive.google.com/drive/folders/1SMQyzJvc6DwJNuhQ_WI8tlCFL5HG2vk6
#    Usage:
#    --amodel HTSAT-tiny
#    --pretrained-audio /path/to/HTSAT-fullset-imagenet-tiny-map=0.467.ckpt

# 2. HTSAT-base:
#    See configuration details in fine_tuning.sh(**recommended**)

# Dataset Configuration Parameters:
# --datasetnames: Dataset identifier
# --datasetpath: Root directory containing dataset
# Download dataset from: https://huggingface.co/datasets/ManzhenWei/MusicSet
#
# Example directory structure:
# /mnt/data/dataset/MusicSet
#
# Usage:
# --datasetnames MusicSet
# --datasetpath /mnt/data/dataset

# Melody Configuration Parameter
# --melody-path: Path to the melody file containing melody information
# Download melody data from: https://huggingface.co/datasets/ManzhenWei/MelodySet
#
# Example directory structure:
# /mnt/data/melody/melody.txt
#
# Usage:
# --melody-path /mnt/data/melody

# For complete parameter documentation, refer to Awesome-Music-Generation/MMGen_train/modules/clmp/training/params.py

CUDA_VISIBLE_DEVICES=0 python -m training.main \
    --save-frequency 20 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --workers=0 \
    --use-bn-sync \
    --warmup 3200 \
    --lr=1e-5 \
    --wd=0.0 \
    --top-k-checkpoint-select-metric="mAP@10" \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --tmodel roberta \
    --amodel HTSAT-tiny \
    --pretrained-audio 'path-to-pretrained-audio' \
    --logs 'path-to-logs' \
    --datasetinfos "train" \
    --datasetnames "your-dataset-name" \
    --datasetpath='path-to-dataset' \
    --melody-path='path-to-melody' \
    --top-k-checkpoint-select-dataset="<datasetnames>-test" \
    --batch-size=32 \
    --epochs=60 \
    # --collect-audio-melody-feature True
