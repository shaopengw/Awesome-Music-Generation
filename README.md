# Awesome-Music-Generation  
![banner](MMGenMMGenBanner.jpg)  
## Welcome to MG<sup>2</sup>!

🎉 We've updated the CLMP training and fine-tuning code and documentation! Come check it out~ 🚀 \[2024-11-09\]

🎉 We've released the [MelodySet](https://huggingface.co/datasets/ManzhenWei/MelodySet) dataset. \[2024-11-08\]

🎉 We've released the [MusicSet](https://huggingface.co/datasets/ManzhenWei/MusicSet) dataset! Come and try it out~ 🎵 \[2024-11-05\]

### Try our demo first! 
&ensp; &rarr; &ensp; 
<a href="https://awesome-mmgen.github.io/"><img src="https://img.shields.io/static/v1?label=Demo! Check out amazing results&message=MG²&color=purple&logo=github.io"></a> &ensp; &larr; &ensp; Click here!

&ensp; &rarr; &ensp; 
<a href="https://arxiv.org/abs/2409.20196"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv.2409&color=red&logo=arXiv"></a> &ensp; &larr; &ensp; Click here!


&ensp; &rarr; &ensp; 
<a href="https://huggingface.co/ManzhenWei/MG2"><img src="https://img.shields.io/static/v1?label=CKPT&message=huggingface&color=yellow&logo=huggingface.co"></a> &ensp; &larr; &ensp; Click here!  

&ensp; &rarr; &ensp; 
<a href="https://huggingface.co/datasets/ManzhenWei/MusicSet"><img src="https://img.shields.io/static/v1?label=Dataset&message=MusicSet&color=green&logo=huggingface.co"></a> &ensp; &larr; &ensp; Click here! 




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
We introduce the newly proposed [MusicSet](https://huggingface.co/datasets/ManzhenWei/MusicSet) dataset, featuring approximately 150,000 high-quality 10-second music-melody-text pairs. 
### Dataset structure of CLMP
We propose CLMP (Contrastive Language-Music Pretraining) to align text description, music waveform and melody before the training of diffusion module. We utilize the [Webdataset](https://github.com/webdataset/webdataset) as a dataloader for music waveform and text description, and we use another dataloader for melody. The MusicSet has been orginized as following for the traning of CLMP:  

```bash
# Ensure that the training data packaged with Webdataset format is orginized as following:
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
### Dataset structure of diffusion module
The dataset structure of diffusion module is as following:

(Noted that you must convert `.flac` files to `.wav` format.)
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
Below is an example of dataset_root.json:
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
Below is an example of train.json:
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
We will release the [MelodySet](https://huggingface.co/datasets/ManzhenWei/MelodySet), containing processed melodies for [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) and [Musicbench](https://huggingface.co/datasets/amaai-lab/MusicBench). We extract the melodies using [basic-pitch](https://github.com/spotify/basic-pitch-ts) and organize them using melody triplets.  MelodySet is a subset of the MusicSet Each waveform file `.wav`  has a corresponding melody file`.txt` with same filename prefix. For example, `00040020.wav` corresponds to `00040020.txt`, and all melodies are placed in a single directory.

The orginization of music waveform and text description are same as that in MusicSet. Thus we only show the dataset structure of melody part as following:
```bash
your_path/
└── melody_text/00040020.txt
               00009570.txt
```

Below is an example of melody, which consists of melody triplets:
```bash
<G4>,<114>,<79>|<A4>,<119>,<81>|<B2>,<159>,<0>|<G4>,<117>,<62>|<A4>,<91>,<77>|<D3>,<202>,<0>|<B4>,<92>,<72>|<A4>,<95>,<77>|<B4>,<98>,<80>|<G3>,<200>,<0>|<A4>,<151>,<30>|<G4>,<95>,<77>|<A4>,<93>,<82>|<F#3>,<146>,<0>|<A2>,<201>,<0>|<G2>,<116>,<117>|<G3>,<149>,<0>|<B2>,<122>,<75>|<D3>,<110>,<77>|<B4>,<206>,<0>|<B4>,<113>,<111>|<B3>,<90>,<95>|<A3>,<110>,<57>|<E5>,<113>,<41>|<G3>,<177>,<0>|<D#5>,<119>,<73>|<B3>,<119>,<32>|<C4>,<108>,<78>|<E5>,<111>,<49>|<F#5>,<117>,<82>|<E5>,<111>,<78>|<F#5>,<114>,<82>|<G3>,<151>,<0>|<G5>,<95>,<73>|<F#5>,<91>,<81>|<G5>,<92>,<78>|<A3>,<143>,<43>|<E4>,<202>,<0>|<F#5>,<152>,<30>|<E5>,<98>,<86>|<D#4>,<139>,<8>|<B3>,<142>,<0>|<F#5>,<94>,<68>|<B3>,<111>,<120>|<G3>,<114>,<84>|<B3>,<118>,<83>|<E3>,<122>,<81>|<G5>,<231>,<0>|<E4>,<234>,<0>|<F#5>,<118>,<63>|<E5>,<114>,<79>|<G3>,<118>,<37>|<D5>,<122>,<76>|<C#5>,<119>,<78>|<E5>,<119>,<77>|<B3>,<100>,<78>|<B4>,<123>,<57>|<E5>,<112>,<71>|<A3>,<209>,<0>|<G5>,<123>,<105>|<A4>,<154>,<0>|<F#5>,<124>,<73>|<A3>,<136>,<22>|<C#4>,<205>,<0>|<E5>,<125>,<28>|<F#5>,<121>,<74>|<A5>,<115>,<72>|<D3>,<144>,<0>|<E3>,<95>,<81>|<E5>,<122>,<62>|<A5>,<115>,<76>|<F#3>,<106>,<84>|<D5>,<117>,<48>|<C5>,<125>,<74>|<D3>,<102>,<74>|<B4>,<120>,<50>|<A4>,<123>,<76>|<B4>,<116>,<80>|<D5>,<117>,<79>|<D4>,<319>,<0>|<A4>,<113>,<65>|<C4>,<114>,<42>|<D5>,<116>,<78>|<B3>,<108>,<84>|<G4>,<114>,<43>
```

## Training and Fine-tuning

Assuming you've gone through the Quick Start guide, let's dive into the training and fine-tuning process! 🚀


```bash
conda activate MMGen_quickstart
```

### CLMP
This section covers the training and fine-tuning process for the CLMP.

```bash
cd your_path/MMGen_train/modules/clmp
```

#### Training
Before running the training script, **review and update** (**crucial**) the paths in [*Awesome-Music-Generation/MMGen_train/modules/clmp/**training.sh***](MMGen_train/modules/clmp/training.sh) as needed. This file contains necessary training details.

```bash
bash training.sh
```
#### Fine-tuning
Similarly, **review and update** (**crucial**) the paths in [*Awesome-Music-Generation/MMGen_train/modules/clmp/**fine_tuning.sh***](MMGen_train/modules/clmp/fine_tuning.sh) before proceeding with fine-tuning.
```bash
bash fine_tuning.sh
```

### CLMP Embedding Extraction and FAISS Index Construction
After CLMP model training or fine-tuning, you'll need to generate embeddings and construct FAISS indices to enable efficient similarity search during the Latent Diffusion training phase. Follow this two-step process:

1. **Generate CLMP Embeddings**
   Enable embedding extraction by adding the following flag to your training configuration:
   ```bash
   --collect-audio-melody-feature True
   ```
   Execute the training or fine-tuning script with this flag:
   ```bash
   bash training.sh  # or fine_tuning.sh
   ```
   The model will generate audio and melody feature embeddings in the following directory:
   ```bash
   your_path/Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing/clmp_embeddings
   ```

2. **Construct FAISS Indices**
   Navigate to the indexing directory and execute the index construction script:
   ```bash
   cd your_path/Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing
   ```
   ```bash
   # you should modify the path of embeddings in this script
   python build_faiss_indices.py 
   ```
   
   The script will generate optimized FAISS indices in:
   ```bash
   your_path/Awesome-Music-Generation/MMGen_train/modules/clmp/faiss_indexing/faiss_indices
   ```

### Diffusion module
Before the training or finetuning of diffusion module, you should prepare required files and replace corresponding file paths in scripts.

First, you should set the mode. In the script `MMGen_train/train/latent_diffusion.py`, for evaluation purpose, please set `only_validation = True`; for training purpose, please set `only_validation = False`.

Then, you should prepare the required files for melody vector database, including `.faiss` and `.npy`, which can be found in [HuggingFace](https://huggingface.co/ManzhenWei/MG2/tree/main). Please replace the path of `.faiss` and `.npy` in script `MMGen_train/modules/latent_diffusion/ddpm.py`

```bash
 # change the melody_npy and melody.faiss to the local path
        melody_npy = np.load("MMGen/melody.npy")
        melody_builder = FaissDatasetBuilder(melody_npy)
        melody_builder.load_index("MMGen/melody.faiss")
```

Afterwards, you can run the following command to train from scratch：
```bash
python3 MMGen_train/train/latent_diffusion.py -c MMGen_train/config/train.yaml
```
Regarding to training dataset, please refer to [Dataset](#dataset) section

### Finetuning of the pretrained model
You can also finetune with our pretrained model, the checkpoint is `mg2-diffusion-checkpoint.ckpt`, which can be found [here](https://huggingface.co/ManzhenWei/MG2/blob/main/mg2-diffusion-checkpoint.ckpt).

Then, you can run the following command to finetune your own model:
```bash
python3 MMGen_train/train/latent_diffusion.py -c MMGen_train/config/train.yaml --reload_from_ckpt data/checkpoints/mg2-diffusion-checkpoint.ckpt
```
Noted that MG<sup>2</sup> is not permitted for commercial use.
## Todo List
- [x] Demo website
- [x] Huggingface checkpoints
- [x] Quick start (Inference)
- [x] Training Datasets
- [x] Training/fine-tuning code
- [x] Online free generation service


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
