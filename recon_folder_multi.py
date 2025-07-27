import argparse
import os
import time
from pathlib import Path
import typing as tp

import torch
import torchaudio
from tqdm import tqdm
from accelerate import Accelerator

from wrapper import build_model


def get_filelist(folder: tp.Union[str, os.PathLike], extensions: tp.Optional[tp.List[str]] = None) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    return [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--input_folder', type=str, default="exp_recon/test-clean")
    parser.add_argument('--model', type=str, default='Encodec')
    parser.add_argument('--output_folder', type=str, default="exp_recon/DUMMY")
    parser.add_argument('--n_quantizers', type=int, default=12)
    parser.add_argument('--target_sr', type=int, default=16000)

    args = parser.parse_args()

    accel = Accelerator()
    rank = accel.process_index
    world_size = accel.num_processes

    os.makedirs(args.output_folder, exist_ok=True)

    # Model & device
    model_device = accel.device

    audio_tokenizer = build_model(args.model, model_device)
    filelist = get_filelist(args.input_folder)
    target_sr = args.target_sr
    sr = audio_tokenizer.sample_rate

    # Split data
    filelist = filelist[rank::world_size]

    if accel.is_main_process:
        print(f"*** model = {args.model} | nq={args.n_quantizers} | sample_rate={audio_tokenizer.sample_rate} ***")

    start_time = time.time()
    with torch.no_grad():
        for file_path in tqdm(filelist, disable=(not accel.is_main_process)):
            wav = audio_tokenizer.load(file_path)
            recon_wav = audio_tokenizer.recon_wav(wav, n_q=args.n_quantizers)
            rel_path = os.path.relpath(file_path, start=args.input_folder)
            
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                resampler = resampler.to(recon_wav.device)
                recon_wav = resampler(recon_wav)


            output_path = os.path.join(args.output_folder, rel_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, recon_wav.cpu(), target_sr)
    end_time = time.time()

    if accel.is_main_process:
        print(f"Total time = {end_time - start_time:.2f} seconds (rank={rank}).")


