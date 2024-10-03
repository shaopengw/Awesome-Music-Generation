from flask import Flask, request, render_template, send_file
import os
import yaml
import torch
from MMGen_train.utilities.tools import build_dataset_json_from_list
from MMGen_train.utilities.model_util import instantiate_from_config
from MMGen_train.utilities.tools import get_restore_step
from MMGen_train.utilities.data.dataset import AudioDataset
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import shutil

app = Flask(__name__)


CONFIG_YAML_PATH = "/mnt/data/wmz/audioldm_train_faiss/audio-ldm-training-finetuning-main_all_run/audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium_wmz_musicbench_ddim_with_rag_1024_new_clap_correct_resume_6Wstep_musiccaps_1wtext_step_resume_only_text_train.yaml"
RELOAD_FROM_CKPT = "/mnt/data/wmz/audioldm_train_faiss/audio-ldm-training-finetuning-main_all_run/data/checkpoints/musiccaps-text-train-origin-1w-checkpoint-fad-133.00-global-step=17799.ckpt"

def infer(dataset_json, configs, config_yaml_path, exp_group_name, exp_name):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])

    log_path = configs["log_directory"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
    )

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    if len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    guidance_scale = configs["model"]["params"]["evaluation_params"][
        "unconditional_guidance_scale"
    ]
    ddim_sampling_steps = configs["model"]["params"]["evaluation_params"][
        "ddim_sampling_steps"
    ]
    n_candidates_per_samples = configs["model"]["params"]["evaluation_params"][
        "n_candidates_per_samples"
    ]

    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    latent_diffusion.generate_sample(
        val_loader,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_gen=n_candidates_per_samples,
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer_endpoint():
    list_inference = request.form['list_inference']

    
    dataset_json = {"data": [{"text": list_inference, "wav": "/mnt/data/wmz/audioldm_train_faiss/audio-ldm-training-finetuning-main_all_run/audioldm_train/web/wav/test.wav"}]}

    assert torch.cuda.is_available(), "CUDA is not available"

    exp_name = os.path.basename(CONFIG_YAML_PATH.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(CONFIG_YAML_PATH))

    config_yaml = yaml.load(open(CONFIG_YAML_PATH, "r"), Loader=yaml.FullLoader)

    config_yaml["reload_from_ckpt"] = RELOAD_FROM_CKPT

    infer(dataset_json, config_yaml, CONFIG_YAML_PATH, exp_group_name, exp_name)
    
    
    generated_paths = infer(dataset_json, config_yaml, CONFIG_YAML_PATH, exp_group_name, exp_name)

    
    generated_wav_path = generated_paths[0]
    
    if not os.path.isfile(generated_wav_path):
        return "Error: Generated WAV file not found.", 404

    return send_file(generated_wav_path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)