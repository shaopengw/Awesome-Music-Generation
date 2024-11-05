# Awesome-Music-Generation  
![banner](MMGenMMGenBanner.jpg)  
## Welcome to MG<sup>2</sup>!
### Try our demo first! 
&ensp; &rarr; &ensp; 
<a href="https://awesome-mmgen.github.io/"><img src="https://img.shields.io/static/v1?label=Demo! Check out amazing results&message=MG²&color=purple&logo=github.io"></a> &ensp; &larr; &ensp; Click here!

&ensp; &rarr; &ensp; 
<a href="https://arxiv.org/abs/2409.20196"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv.2409&color=red&logo=arXiv"></a> &ensp; &larr; &ensp; Click here!


&ensp; &rarr; &ensp; 
<a href="https://huggingface.co/ManzhenWei/MG2"><img src="https://img.shields.io/static/v1?label=CKPT&message=huggingface&color=yellow&logo=huggingface.co"></a> &ensp; &larr; &ensp; Click here!  
This repository contains the implementation of the music generation model **MG<sup>2</sup>**, the first novel approach using melody to guide the music generation that, despite a pretty simple method and extremely limited resources, achieves excellent performance.

Anyone can use this model to generate personalized background music for their short videos on platforms like TikTok, YouTube Shorts, and Meta Reels. Additionally, it is very cost-effective to fine-tune the model with your own private music dataset.

## Video
You can watch the introduction video on  

&ensp; &rarr; &ensp; 
<a href="https://youtu.be/kc2n-ByWB-M?si=U-gjvAuBjD-HOtS7"><img src="https://img.shields.io/static/v1?label=Watch&message=YouTube&color=red&logo=youtube"></a> &ensp; &larr; &ensp; Click here!

&ensp; &rarr; &ensp; 
<a href="https://www.bilibili.com/video/BV1K84FeBEqo/?share_source=copy_web&vd_source=d808713ed70be6b862e6ccbcb28d2f5b"><img src="https://img.shields.io/static/v1?label=Watch&message=Bilibili&color=blue&logo=bilibili"></a> &ensp; &larr; &ensp; Click here!

## Online Service

Now you can try music generation with your own prompt on our   

&ensp; &rarr; &ensp; 
<a href="https://mg2.vip.cpolar.cn/"><img src="https://img.shields.io/static/v1?label=Try it&message=MG² Website&color=green&logo=googlechrome"></a> &ensp; &larr; &ensp; Click here!

**Tips**: To generate high-quality music using MG<sup>2</sup>, you would want to craft detailed and descriptive prompts that provide rich context and specific musical elements.


## Quick Start

To get started with **MG<sup>2</sup>**, follow the steps below:

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

### Step 3: Download checkpoints from [huggingface]([https://huggingface.co/ManzhenWei/MMGen](https://huggingface.co/ManzhenWei/MG2))
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
## Dataset
### MusicSet
We introduce the new MusicSet dataset, featuring approximately 150,000 high-quality 10-second music-text pairs. The dataset can be accessed at the following URL: https://huggingface.co/datasets/ManzhenWei/MusicSet
### CLMP training dataset format
In the training of CLMP, it is necessary to align and train the three dimensions of audio, text-description, and melody-text. We utilize the webdataset format for the training data of audio and text-description, and separately load the melody-text using a dataloader. Within the [MusicSet](https://huggingface.co/datasets/ManzhenWei/MusicSet) dataset, we have already processed the data into the webdataset format. If you wish to use your own training data and package it into the webdataset format, please refer to the instructions at the following link: https://github.com/webdataset/webdataset

```bash
# Ensure that the training data packaged in the webdataset format is placed in the following directory
clmp/
└── dataset/
    └── MusicSet/
        └──train/pretrain0.tar
                 pretrain1.tar
                 pretrain2.tar
                 ...
        └──valid/
        └──test/
```
### MG<sup>2</sup> of diffusion training dataset format
When training MG<sup>2</sup>, it is necessary to convert `.flac` files to `.wav` format.
```bash
Awesome-Music-Generation/
└── data/
    └── dataset/
       └── audioset/
           └── wav/00040020.wav
                   00009570.wav
                   ...
       └── metadata/dataset_root.json
           └── MusicSet/
               └── datafiles/train.json
                             valid.json
                             test.json
```
Here is an example content for dataset_root.json:
```bash
{
    "MusicSet": "/mnt/data/wmz/Awesome-Music-Generation/data/dataset/audioset",
    "comments": {},
    "metadata": {
      "path": {
        "MusicSet": {
          "train": "./data/dataset/metadata/MusicSet/datafiles/train.json",
          "test": "./data/dataset/metadata/MusicSet/datafiles/test.json",
          "val": "./data/dataset/metadata/MusicSet/datafiles/valid.json",
          "class_label_indices": ""
        }
      }
    }
  }
```
Here is an example content for train.json:
```bash
{
    "data": [
        {
            "wav": "wav/00040020.wav",
            "seg_label": "",
            "labels": "",
            "caption": " The song starts with the high and fuzzy tone of an alarm bell beeping until a button is pressed, which triggers the grungy sound of an electric guitar being played in a rock style.", "The beat then counts to four, enhancing the overall rhythm."
        },
        {
            "wav": "wav/00009570.wav",
            "seg_label": "",
            "labels": "",
            "caption": "This lively song features a male vocalist singing humorous lyrics over a medium-fast tempo of 106.", "0 beats per minute.", "Accompanied by keyboard harmony, acoustic guitar, steady drumming, and simple bass lines, the catchy tune is easy to sing along with.", "Set in the key of B major, the chord sequence includes Abm7, F#/G#, and Emaj7.", "With its spirited and animated feel, this fun track is sure to keep listeners engaged from start to finish."
        }
    ]
}
```
### MelodySet
We have created the MelodySet dataset. We extracted the melody using [basic_pitch](https://github.com/spotify/basic-pitch-ts) and organized it into a format of melody triplets, with details available in the paper. Each `.wav` file has a corresponding melody text`.txt`, for example, `00040020.wav` corresponds to `00040020.txt`, and all melody texts are placed in a single directory.
```bash
your_path/
└── melody_text/00040020.txt
               00009570.txt
```

Here is an example content for `.txt`
```bash
<G4>,<114>,<79>|<A4>,<119>,<81>|<B2>,<159>,<0>|<G4>,<117>,<62>|<A4>,<91>,<77>|<D3>,<202>,<0>|<B4>,<92>,<72>|<A4>,<95>,<77>|<B4>,<98>,<80>|<G3>,<200>,<0>|<A4>,<151>,<30>|<G4>,<95>,<77>|<A4>,<93>,<82>|<F#3>,<146>,<0>|<A2>,<201>,<0>|<G2>,<116>,<117>|<G3>,<149>,<0>|<B2>,<122>,<75>|<D3>,<110>,<77>|<B4>,<206>,<0>|<B4>,<113>,<111>|<B3>,<90>,<95>|<A3>,<110>,<57>|<E5>,<113>,<41>|<G3>,<177>,<0>|<D#5>,<119>,<73>|<B3>,<119>,<32>|<C4>,<108>,<78>|<E5>,<111>,<49>|<F#5>,<117>,<82>|<E5>,<111>,<78>|<F#5>,<114>,<82>|<G3>,<151>,<0>|<G5>,<95>,<73>|<F#5>,<91>,<81>|<G5>,<92>,<78>|<A3>,<143>,<43>|<E4>,<202>,<0>|<F#5>,<152>,<30>|<E5>,<98>,<86>|<D#4>,<139>,<8>|<B3>,<142>,<0>|<F#5>,<94>,<68>|<B3>,<111>,<120>|<G3>,<114>,<84>|<B3>,<118>,<83>|<E3>,<122>,<81>|<G5>,<231>,<0>|<E4>,<234>,<0>|<F#5>,<118>,<63>|<E5>,<114>,<79>|<G3>,<118>,<37>|<D5>,<122>,<76>|<C#5>,<119>,<78>|<E5>,<119>,<77>|<B3>,<100>,<78>|<B4>,<123>,<57>|<E5>,<112>,<71>|<A3>,<209>,<0>|<G5>,<123>,<105>|<A4>,<154>,<0>|<F#5>,<124>,<73>|<A3>,<136>,<22>|<C#4>,<205>,<0>|<E5>,<125>,<28>|<F#5>,<121>,<74>|<A5>,<115>,<72>|<D3>,<144>,<0>|<E3>,<95>,<81>|<E5>,<122>,<62>|<A5>,<115>,<76>|<F#3>,<106>,<84>|<D5>,<117>,<48>|<C5>,<125>,<74>|<D3>,<102>,<74>|<B4>,<120>,<50>|<A4>,<123>,<76>|<B4>,<116>,<80>|<D5>,<117>,<79>|<D4>,<319>,<0>|<A4>,<113>,<65>|<C4>,<114>,<42>|<D5>,<116>,<78>|<B3>,<108>,<84>|<G4>,<114>,<43>
```
We will release the complete MelodySet dataset in the near future.
## Todo List
- [x] Demo website
- [x] Huggingface checkpoints
- [x] Quick start (Inference)
- [x] Training Datasets
- [ ] Training/fine-tuning code
- [x] Online free generation service
- [ ] Checkpoints on larger datasets

---
### Feel free to explore the repository and contribute!
---
## Acknowledgement

We sincerely acknowledge the developers of the following open-source code bases. These resources are invaluable sparks that ignite innovation and progress in the real world 🎆!

- [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [https://github.com/haoheliu/AudioLDM-training-finetuning](https://github.com/haoheliu/AudioLDM-training-finetuning)
- [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
- [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- [https://mtg.github.io/mtg-jamendo-dataset](https://mtg.github.io/mtg-jamendo-dataset)
>The research is supported by the Key Technologies Research and Development Program under Grant No. 2020YFC0832702, and National Natural Science Foundation of China under Grant Nos. 71910107002, 62376227, 61906159, 62302400, 62176014, and Sichuan Science and Technology Program under Grant No. 2023NSFSC0032, 2023NSFSC0114, and Guanghua Talent Project of Southwestern University of Finance and Economics.

## Citation

```bibtex
@article{wei2024melodyneedmusicgeneration,
      title={Melody Is All You Need For Music Generation}, 
      author={Shaopeng Wei and Manzhen Wei and Haoyu Wang and Yu Zhao and Gang Kou},
      year={2024},
      eprint={2409.20196},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2409.20196}, 
}
