import torchaudio
import torch

import sys
sys.path.append("../")
from WavTokenizer.encoder.utils import convert_audio
from WavTokenizer.decoder.pretrained import WavTokenizer


device=torch.device('cpu')

from pathlib import Path
audio_path = "LJ001-0001.wav"
audio_path = "2001000001.wav"

save_path = f"WavTokenizer_{Path(audio_path).stem}.wav"

# config_path = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# model_path = "WavTokenizer_small_320_24k_4096.ckpt"

# config_path = "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# model_path = "WavTokenizer_small_600_24k_4096.ckpt"

# config_path = "/home/jiahelei/DATA/pretrained_models/Wav_tokenizer/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# model_path = "/home/jiahelei/DATA/pretrained_models/Wav_tokenizer/WavTokenizer_small_600_24k_4096.ckpt"

config_path = "../WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/home/jiahelei/DATA/pretrained_models/Wav_tokenizer/wavtokenizer_large_unify_600_24k.ckpt"
save_path = f"WavTokenizer_0.52k_{Path(audio_path).stem}.wav"

# config_path = "../WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# model_path = "/home/jiahelei/DATA/pretrained_models/Wav_tokenizer/wavtokenizer_large_speech_320_24k.ckpt"
# save_path = f"WavTokenizer_0.975k_{Path(audio_path).stem}.wav"


# wavtokenizer_large_unify_600_24k.ckpt

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)

# model_name = "novateur/WavTokenizer-medium-music-audio-75token"
# wavtokenizer = WavTokenizer.from_pretrained(repo_id = model_name)

wavtokenizer = wavtokenizer.to(device)


wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1) 
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id) 
torchaudio.save(save_path, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)