import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.utils.checkpoint as checkpoint
import pretty_midi
import random

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from itertools import repeat
from .utils import do_mixup, interpolate

from .feature_fusion import iAFF, AFF, DAF

from transformers import AutoTokenizer
import numpy as np
import torch
import json
from transformers import AutoModel, AutoConfig

class MelodyPooler(nn.Module):
    def __init__(self, hidden_size = 4096):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        averaged_pool = torch.mean(hidden_states, dim=1)
        pooled_output = self.dense(averaged_pool)
        # pooled_output = self.activation(pooled_output)
        return pooled_output

def get_tokenizer_embedding_weight(model_directory):
        
        index_file = f"{model_directory}/pytorch_model.bin.index.json"
        with open(index_file, "r") as f:
            index_data = json.load(f)

        weight_map = index_data["weight_map"]

        
        embedding_weight_key = "model.tok_embeddings.weight"
        embedding_weight_file = weight_map[embedding_weight_key]

        
        embedding_weight_path = f"{model_directory}/{embedding_weight_file}"
        state_dict = torch.load(embedding_weight_path)
        
        config = AutoConfig.from_pretrained(model_directory, trust_remote_code=True)

        
        model = AutoModel.from_config(config, trust_remote_code=True)
        
        vocab_embedding_weight = model.get_input_embeddings().weight.data.copy_(state_dict[embedding_weight_key])
        return vocab_embedding_weight



def text_to_indexed_segments(json_file_path, text):
    """
    Convert text to indexed sequence

    Args:
        json_file_path (str): Path to JSON file
        text (str): Input text

    Returns:
        list: Sequence of indices
    """
    
    with open(json_file_path, 'r') as f:
        vocab = json.load(f)

    
    if "," not in vocab or "|" not in vocab:
        raise KeyError("JSON file missing comma or vertical bar index")

    
    segments = text.split('|')

    
    indexed_segments = []
    for segment in segments:
        if segment:
            tokens = segment.split(',')
            indexed_tokens = []
            for token in tokens:
                if token in vocab:
                    indexed_tokens.append(vocab[token])
                else:
                    
                    indexed_tokens.append(vocab["<unk>"])
            
            for i in range(1, len(indexed_tokens)):
                indexed_tokens.insert(i * 2 - 1, vocab[","])  
            indexed_segments.append(indexed_tokens)

    
    final_indexed_segments = []
    for i, segment in enumerate(indexed_segments):
        final_indexed_segments.extend(segment)
        if i < len(indexed_segments) - 1:
            final_indexed_segments.append(vocab["|"])  

    
    final_indexed_segments.insert(0, 0)

    return final_indexed_segments

class Melody(nn.Module):
    def __init__(self, 
                 model_path = '/mnt/data/wmz/audioldm_clap_repair/AudioLDM-training-finetuning-main/audioldm_train/modules/clap/songcomposer/melody_encoder/melody_embedding_weights.pt', 
                 tokenizer_path = '/mnt/data/wmz/audioldm_clap_repair/AudioLDM-training-finetuning-main/audioldm_train/modules/clap/songcomposer/melody_encoder_backup'):
                 
        super(Melody, self).__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.mlp_1024_to_768 = nn.Linear(1024, 768)
        self.LayerNorm = nn.LayerNorm(1024, eps=1e-5)
        self.pooler = MelodyPooler(hidden_size=1024)

        
        self.melody_embbedding_weight = nn.Embedding(655, 1024)  
        # self.melody_embbedding_weight.weight.requires_grad = False

        
        
        # self.melody_embbedding_weight.weight.data = self.melody_embbedding_weight_songcomposer.data
        
        # self.melody_embbedding_weight.weight = nn.Parameter(melody_embbedding_weight_songcomposer)
        
        # data1 = self.melody_embbedding_weight_songcomposer.data[1]
        # data2 = self.melody_embbedding_weight.weight.data[1]
        # if torch.equal(data1, data2):
        #     print("The rows are equal.")
        # else:
        #     print("The rows are not equal.")
        
        # self.melody_embbedding_weight.weight = nn.Parameter(torch.Tensor(self.melody_embbedding_weight_songcomposer.size()), requires_grad=True)
        # self.melody_embbedding_weight.weight.data.copy_(self.melody_embbedding_weight_songcomposer)
        
        
        self.fc1 = nn.Linear(768, 768)
        self.activation1 = nn.GELU()  
        self.activation_add = nn.GELU()
        # self.fc2 = nn.Linear(768, 768)
        
    
    def forward(self, melody_texts, device):
        all_token_embeddings = []
        # print("melody_texts check", melody_texts[0]) # 
        for melody_text in melody_texts:
            # tokens = self.tokenizer(melody_text)['input_ids']
            tokens = text_to_indexed_segments('/mnt/data/wmz/audioldm_clap_repair/AudioLDM-training-finetuning-main/audioldm_train/modules/clap/songcomposer/only_index/re_index.json', melody_text)
            tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)            
            token_embeddings = self.melody_embbedding_weight(tokens_tensor)
            token_embeddings = token_embeddings.to(torch.float32) 
            # print("token_embeddings shape1111", token_embeddings.shape) # [xxx, 1024]
            # import sys
            # sys.exit()
            target_shape = (1000, 1024)
            
            if token_embeddings.shape[0] < target_shape[0]:
                
                padding = torch.zeros(target_shape[0] - token_embeddings.shape[0], token_embeddings.shape[1], device=token_embeddings.device)
                token_embeddings = torch.cat((token_embeddings, padding), dim=0)
            elif token_embeddings.shape[0] > target_shape[0]:
                
                token_embeddings = token_embeddings[:target_shape[0], :]
            all_token_embeddings.append(token_embeddings) 
        
        
        all_token_embeddings = torch.stack(all_token_embeddings) 
        
        all_token_embeddings = self.pooler(all_token_embeddings)

        all_token_embeddings = self.LayerNorm(all_token_embeddings)
        
        token_embeddings = self.mlp_1024_to_768(all_token_embeddings)

        
        token_embeddings = self.activation_add(token_embeddings)

        
        token_embeddings = self.fc1(token_embeddings)
        token_embeddings = self.activation1(token_embeddings)
        # token_embeddings = self.fc2(token_embeddings)
        # token_embeddings = self.activation2(token_embeddings)
        return token_embeddings 
    
    
