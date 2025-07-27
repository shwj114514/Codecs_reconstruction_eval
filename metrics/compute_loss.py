import os
import argparse
import glob
import librosa
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 如果你的 MultiScaleSTFTLoss、MelSpectrogramLoss 定义在别的文件，请根据实际位置修改
from losses import MultiScaleSTFTLoss, MelSpectrogramLoss

# 如果有 GPU 并想用 GPU，就写成下面这样
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 你实际需要支持的音频格式
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]

# 定义仅需要的2个 Loss
stft_loss = MultiScaleSTFTLoss().to(DEVICE)
mel_loss = MelSpectrogramLoss().to(DEVICE)

def get_metrics(ref_path, gen_path, sr=16000):
    """
    读取参考音频 ref_path 和生成音频 gen_path，
    只计算 stft_loss & mel_loss，并返回它们的数值。
    """
    # 读音频
    ref_wav, _ = librosa.load(ref_path, sr=sr)
    gen_wav, _ = librosa.load(gen_path, sr=sr)

    # 对齐长度
    min_len = min(len(ref_wav), len(gen_wav))
    ref_wav = ref_wav[:min_len]
    gen_wav = gen_wav[:min_len]

    # 转成张量 [B, 1, T]
    ref_tensor = torch.from_numpy(ref_wav).unsqueeze(0).unsqueeze(1).to(DEVICE)
    gen_tensor = torch.from_numpy(gen_wav).unsqueeze(0).unsqueeze(1).to(DEVICE)

    # 计算 STFT Loss & Mel Loss
    stft_val = stft_loss(ref_tensor, gen_tensor).item()
    mel_val = mel_loss(ref_tensor, gen_tensor).item()

    return stft_val, mel_val

def compute_similarity(ref_dir, gen_dir):
    """
    遍历 ref_dir (参考音频) 与 gen_dir (生成音频)，
    找到同名文件对后依次计算 STFT Loss、Mel Loss，
    最后打印它们的平均值。
    """
    ref_dir = Path(ref_dir)
    gen_dir = Path(gen_dir)

    # 收集参考音频文件
    ref_files = []
    for ext in AUDIO_EXTENSIONS:
        ref_files.extend(ref_dir.rglob(f"*{ext}"))
    ref_files = sorted(ref_files)

    # 收集生成音频文件，做一个 {stem: 文件Path} 的字典
    gen_dict = {}
    gen_files = []
    for ext in AUDIO_EXTENSIONS:
        gen_files.extend(gen_dir.rglob(f"*{ext}"))
    for g in gen_files:
        gen_dict[g.stem] = g

    stft_vals = []
    mel_vals = []

    # 遍历所有参考文件，匹配同名文件
    for r in tqdm(ref_files, desc="Calculating metrics"):
        stem = r.stem
        if stem in gen_dict:
            g = gen_dict[stem]
            stft_v, mel_v = get_metrics(str(r), str(g), sr=16000)
            stft_vals.append(stft_v)
            mel_vals.append(mel_v)
        else:
            # 如果没找到匹配文件，可以选择性地打印个提示
            print(f"[Warning] No matching file for: {r.name}")

    # 只在找到了匹配对的情况下计算平均
    if len(stft_vals) > 0:
        avg_stft = sum(stft_vals) / len(stft_vals)
        avg_mel = sum(mel_vals) / len(mel_vals)
        print("\n===== Final Averages =====")
        print(f"Average STFT Loss: {avg_stft:.6f}")
        print(f"Average Mel Loss:  {avg_mel:.6f}")
    else:
        print("No matched audio pairs found. Nothing to compute.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute STFT & Mel Loss between reference and generated audio.")
    parser.add_argument('-r', '--ref_dir', required=True, help="Reference folder containing audio files")
    parser.add_argument('-g', '--gen_dir', required=True, help="Generated folder containing audio files")

    args = parser.parse_args()
    compute_similarity(args.ref_dir, args.gen_dir)