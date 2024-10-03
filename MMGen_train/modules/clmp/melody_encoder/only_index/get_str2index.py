import json


json_file_path = '/mnt/data/wmz/audioldm_clap_repair/AudioLDM-training-finetuning-main/audioldm_train/modules/clap/songcomposer/only_index/re_index.json'


with open(json_file_path, 'r') as f:
    vocab = json.load(f)


if "," not in vocab or "|" not in vocab:
    raise KeyError("JSON 文件中缺少逗号或竖杠的索引")


# text = "<E4>,<154>,<79>|<E4>,<134>,<88>|<E4>,<137>,<88>|"
text = "<E2>,<117>,<79>|<E2>,<115>,<79>|<E4>,<117>,<79>|<E4>,<117>,<99>|<E4>,<108>,<79>|<B2>,<119>,<33>|<B2>,<108>,<79>|<B2>,<124>,<79>|<B3>,<124>,<23>|<B2>,<112>,<79>|<E2>,<111>,<119>|<E3>,<106>,<51>|<C#4>,<126>,<89>|<B2>,<117>,<112>|<B2>,<112>,<79>|<B2>,<110>,<126>|<A#3>,<110>,<82>|<E2>,<108>,<94>|<B3>,<108>,<55>|<B2>,<113>,<158>|<B2>,<112>,<79>|<B2>,<115>,<119>|<E2>,<108>,<79>|<E2>,<125>,<79>|<C#2>,<123>,<160>|<A1>,<140>,<164>|<E3>,<110>,<183>|<F#4>,<131>,<117>|<F#4>,<146>,<79>|<F#4>,<158>,<79>|<E4>,<113>,<101>|<C#4>,<108>,<82>|<C#4>,<115>,<79>|<C4>,<108>,<79>|<C4>,<117>,<79>|<E2>,<125>,<27>|<B3>,<110>,<94>|<E2>,<138>,<27>|<B3>,<110>,<57>"


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

print("Indexed Segments:", final_indexed_segments)