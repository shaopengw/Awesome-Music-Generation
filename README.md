# Awesome-Music-Generation

Welcome to MMGen!

This repository contains the implementation of the music generation model **MMGen**, the first novel approach using melody to guide the music generation that, despite a pretty simple method and extremely limited resources, achieves excellent performance.

Anyone can use this model to generate personalized background music for their short videos on platforms like TikTok, YouTube Shorts, and Meta Reels. Additionally, it is very cost-effective to fine-tune the model with your own private music dataset.

## Demo
Check out our live demo at [https://awesome-mmgen.github.io/](https://awesome-mmgen.github.io/).

## Paper
Read our research paper on [arXiv](https://arxiv.org/abs/2409.20196).

## Quick Start

To get started with **MMGen**, follow the steps below:

### Step 1: Clone the repository
```bash
git clone https://github.com/shaopengw/Awesome-Music-Generation.git
cd Awesome-Music-Generation
```

### Step 2: Set up the Conda environment
```bash
# Create and activate the environment from the provided environment file
conda env create -f environment.yml
conda activate MMGen_quickstart
```

### Step 3: Download checkpoints from [\[huggingface\](https://huggingface.co/ManzhenWei/MMGen)](https://huggingface.co/ManzhenWei/MMGen)
```bash
# Ensure that the checkpoints are stored in the following directory structure
Awesome-Music-Generation/
└── data/
    └── checkpoints/
```

### Step 4: Modify the PYTHONPATH environment variables in the quick_start.sh script
```bash
# Update the paths to reflect your local environment setup
# Replace:
export PYTHONPATH=/mnt/sda/quick_start_demonstration/Awesome-Music-Generation:$PYTHONPATH
export PYTHONPATH=/mnt/sda/quick_start_demonstration/Awesome-Music-Generation/data:$PYTHONPATH
# With:
export PYTHONPATH=/your/local/path/Awesome-Music-Generation:$PYTHONPATH
export PYTHONPATH=/your/local/path/Awesome-Music-Generation/data:$PYTHONPATH
```

### Step 5: Assign execution permissions for the script
```bash
chmod +x quick_start.sh
```
### Step 6: Execute the quick start script
```bash
bash quick_start.sh
```
### Allow the script to run for several minutes. Upon completion, the results will be available in the following directory:
```bash
Awesome-Music-Generation/log/latent_diffusion/quick_start/quick_start
```


## Checkpoints
https://huggingface.co/ManzhenWei/MMGen

## Todo List
- [x] Demo website
- [x] Huggingface checkpoints
- [x] Quick start (Inference)
- [ ] Training Datasets
- [ ] Training/fine-tuning code
- [ ] Online free generation service
- [ ] Checkpoints on larger datasets



---

Feel free to explore the repository and contribute!

## Citation

```bibtex
@misc{wei2024melodyneedmusicgeneration,
      title={Melody Is All You Need For Music Generation}, 
      author={Shaopeng Wei and Manzhen Wei and Haoyu Wang and Yu Zhao and Gang Kou},
      year={2024},
      eprint={2409.20196},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2409.20196}, 
}
