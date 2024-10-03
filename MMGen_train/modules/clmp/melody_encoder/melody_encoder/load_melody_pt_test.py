import torch
from torch import nn
from transformers import AutoTokenizer

model_path = ""
model = torch.load(model_path)
print(model.shape)


model_directory = '/mnt/sda/upload_github/Awesome-Music-Generation/MMGen_train/modules/clmp/melody_encoder/melody_encoder'
tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=True)



# self.tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=True)


# token = 1

# word = tokenizer.convert_ids_to_tokens(token)
# print(f"token: {token}")



# midi = prompt
# prompt = "<E4>,<154>,<79>|<E4>,<134>,<88>|<E4>,<137>,<88>|<F#4>,<151>,<79>|<E4>,<154>,<79>|<D#4>,<154>,<79>|<C#4>,<157>,<79>|<B3>,<172>,<79>|<E4>,<151>,<127>|<E4>,<137>,<88>|<E4>,<137>,<88>|<F#4>,<151>,<79>|<E4>,<151>,<79>|<D#4>,<160>,<79>|<C#4>,<157>,<79>|<B3>,<151>,<79>|<G#3>,<137>,<79>|<B3>,<151>,<79>|<G#3>,<189>,<79>|<F#3>,<157>,<79>|<G#3>,<137>,<79>|<G#3>,<147>,<79>|<F#3>,<144>,<79>|<E3>,<151>,<79>|<F#3>,<141>,<79>|<G#3>,<166>,<79>|<B3>,<219>,<79>|<E4>,<154>,<160>|<E4>,<130>,<88>|<E4>,<144>,<88>|<F#4>,<147>,<79>|<E4>,<157>,<79>|<D#4>,<154>,<79>|<C#4>,<151>,<79>|<B3>,<118>,<79>|<B3>,<118>,<79>|<G#3>,<207>,<79>|<B3>,<205>,<79>|<G#3>,<205>,<79>"
# prompt = "<B4>,<100>,<79>|<G3>,<123>,<98>|<B3>,<114>,<72>|<D4>,<104>,<80>|<B4>,<170>,<0>|<C5>,<88>,<78>|<G4>,<119>,<74>|<D5>,<103>,<75>|<C5>,<94>,<65>|<D5>,<97>,<77>|<G4>,<102>,<77>|<F#4>,<116>,<90>|<C5>,<149>,<0>|<A3>,<175>,<0>|<B4>,<93>,<79>|<C5>,<97>,<79>|<E4>,<121>,<52>|<D4>,<124>,<105>|<F#4>,<118>,<72>|<B3>,<198>,<0>|<D5>,<206>,<0>|<B4>,<110>,<80>|<D5>,<117>,<83>|<B3>,<112>,<79>|<F#5>,<125>,<38>|<G5>,<121>,<72>|<C4>,<110>,<75>|<E5>,<114>,<40>|<G5>,<114>,<81>|<D5>,<113>,<79>|<B3>,<116>,<40>|<G5>,<120>,<77>|<C5>,<115>,<78>|<B4>,<116>,<81>|<A3>,<152>,<0>|<G4>,<329>,<0>|<C5>,<116>,<66>|<F#4>,<144>,<37>|<A5>,<119>,<37>|<C5>,<113>,<79>|<B4>,<118>,<82>|<B3>,<89>,<83>|<A4>,<112>,<66>|<G3>,<200>,<0>|<G4>,<122>,<34>|<D4>,<127>,<78>|<F#3>,<115>,<64>|<G4>,<86>,<80>|<E3>,<104>,<72>|<B4>,<200>,<0>|<D3>,<122>,<26>|<E4>,<126>,<82>|<E5>,<115>,<65>|<E3>,<90>,<81>|<A4>,<116>,<67>|<E5>,<118>,<83>|<C#3>,<208>,<0>|<G4>,<113>,<69>|<F#4>,<114>,<83>|<C#3>,<102>,<80>|<G4>,<114>,<56>|<E5>,<119>,<78>|<G4>,<113>,<79>|<A3>,<248>,<0>|<F#4>,<130>,<19>|<F#3>,<102>,<70>|<E4>,<114>,<51>|<D3>,<203>,<0>|<D4>,<126>,<29>|<E4>,<121>,<70>|<F#4>,<124>,<70>|<C3>,<106>,<71>|<G4>,<125>,<48>" 
# with open(melody_text_path, 'r', encoding='utf-8') as f:
#     content = f.read()
content = "<E4>"

print("content:", content)

embedding_weight = model
prompt = content

tokens = tokenizer(prompt)['input_ids']
print("========================================")
print("tokens:", tokens, len(tokens))
print("========================================")

tokens_tensor = torch.tensor(tokens).unsqueeze(0)

token_embeddings = embedding_weight[tokens]

print(token_embeddings, token_embeddings.shape)
