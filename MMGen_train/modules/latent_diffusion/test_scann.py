import numpy as np
music_features = np.load('/work/home/acdfqcivpi/AudioLDM-training-finetuning-main_all_run/audioldm_train/modules/latent_diffusion/scann_epoch.npy')
print(music_features[0].shape)
print(music_features.shape)