import os
import argparse
import librosa
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]

def visqol(ref, est, mode="speech"):
    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
        config.audio.sample_rate = 48000
    else:
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
        config.audio.sample_rate = 16000

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )
    api = visqol_lib_py.VisqolApi()
    api.Create(config)
    return api.Measure(ref.astype(float), est.astype(float)).moslqo

def get_visqol_score(ref_path, gen_path, sr=16000):
    ref_wav, _ = librosa.load(ref_path, sr=sr)
    gen_wav, _ = librosa.load(gen_path, sr=sr)
    min_len = min(len(ref_wav), len(gen_wav))
    ref_wav = ref_wav[:min_len]
    gen_wav = gen_wav[:min_len]
    return visqol(ref_wav, gen_wav, "speech")

def compute_similarity(ref_dir, gen_dir):
    ref_dir = Path(ref_dir)
    gen_dir = Path(gen_dir)
    ref_files = []
    for ext in AUDIO_EXTENSIONS:
        ref_files.extend(ref_dir.rglob(f"*{ext}"))
    ref_files = sorted(ref_files)

    gen_dict = {}
    gen_files = []
    for ext in AUDIO_EXTENSIONS:
        gen_files.extend(gen_dir.rglob(f"*{ext}"))
    for g in gen_files:
        gen_dict[g.stem] = g

    scores = []
    for r in tqdm(ref_files):
        if r.stem in gen_dict:
            score = get_visqol_score(str(r), str(gen_dict[r.stem]))
            scores.append(score)
        else:
            print(f"[Warning] No matching file for: {r.name}")

    if scores:
        print(f"Average ViSQOL: {sum(scores)/len(scores):.6f}")
    else:
        print("No matched audio pairs found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_dir', required=True)
    parser.add_argument('-g', '--gen_dir', required=True)
    args = parser.parse_args()
    compute_similarity(args.ref_dir, args.gen_dir)