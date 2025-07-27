import torch
import torchaudio
from stable_codec import StableCodec

from pathlib import Path
audio_path = "LJ001-0001.wav"
# audio_path = "2001000001.wav"

save_path = f"StableCodec_{Path(audio_path).stem}.wav"
device = "cuda:5"

model = StableCodec(
    model_config_path="/home/jiahelei/DATA/pretrained_models/StableCodec/model_config.json",
    ckpt_path="/home/jiahelei/DATA/pretrained_models/StableCodec/model.safetensors", # optional, can be `None`,
    device = torch.device(device)
)


latents, tokens = model.encode(audio_path)
decoded_audio = model.decode(tokens).detach().cpu().squeeze(0)
torchaudio.save(save_path, decoded_audio, model.sample_rate)