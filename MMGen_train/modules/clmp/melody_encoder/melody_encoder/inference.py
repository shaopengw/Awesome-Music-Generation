from transformers import AutoTokenizer, AutoModel
ckpt_path = "Mar2Ding/songcomposer_sft"

tokenizer_path = "/mnt/sda/upload_github/Awesome-Music-Generation/MMGen_train/modules/clmp/melody_encoder/melody_encoder"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True).cuda().half()
prompt = 'Create a song on brave and sacrificing with a rapid pace.'
model.inference(prompt, tokenizer)
