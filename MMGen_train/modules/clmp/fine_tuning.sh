#!/bin/bash
# Detailed Description of Key Parameters:

# Model Configuration Parameters
# --tmodel: Text encoder backbone (default: roberta)
# --amodel: Audio encoder backbone (HTSAT-base)
# --resume: Path to pre-trained model checkpoint

# We provide a checkpoint for fine-tuning:
# https://huggingface.co/ManzhenWei/MG2/blob/main/mg2-clmp.pt
# Usage:
# --amodel HTSAT-base
# --resume path/to/mg2-clmp.pt

# Dataset Configuration Parameters:
# --datasetnames: Dataset identifier
# --datasetpath: Root directory containing dataset
# download dataset from: https://huggingface.co/datasets/ManzhenWei/MusicSet
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
    --datasetpath='/mnt/data/melody_all/webdataset_tar/dataset' \
    --melody-path='/mnt/data/melody_all/melody_text' \
    --precision="fp32" \
    --batch-size=32 \
    --lr=1e-5 \
    --wd=0.0 \
    --epochs=60 \
    --workers=0 \
    --use-bn-sync \
    --amodel HTSAT-base \
    --tmodel roberta \
    --warmup 3200 \
    --datasetnames "your-dataset-name" \
    --datasetinfos "train" \
    --top-k-checkpoint-select-dataset="<datasetnames>-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --logs 'path-to-logs' \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --resume 'your-path/mg2-clmp.pt' \
    # --collect-audio-melody-feature True
