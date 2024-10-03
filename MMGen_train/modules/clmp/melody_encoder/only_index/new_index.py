import json


with open('/mnt/data/wmz/audioldm_clap_repair/AudioLDM-training-finetuning-main/audioldm_train/modules/clap/songcomposer/only_index/added_tokens.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


new_data = {key: idx for idx, key in enumerate(data.keys())}


with open('/mnt/data/wmz/audioldm_clap_repair/AudioLDM-training-finetuning-main/audioldm_train/modules/clap/songcomposer/only_index/re_index.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

