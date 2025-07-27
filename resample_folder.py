import argparse
import os
import time
from pathlib import Path
import typing as tp

import torch
import torchaudio
from tqdm import tqdm


def get_filelist(folder: tp.Union[str, os.PathLike], extensions: tp.Optional[tp.List[str]] = None) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)

    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    return [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]


if __name__ == "__main__":
    INPUT_FOLDER = "exp_recon/test-clean"
    TARGET_SR = 16000
    # TARGET_SR = 24000

    OUTPUT_FOLDER = f"{INPUT_FOLDER}_{TARGET_SR}"

    parser = argparse.ArgumentParser(description="Resample audio files in a folder to a specified sample rate.")
    parser.add_argument('--device', type=str, default='cpu', help='Device to run resampling on')
    parser.add_argument('--input_folder', type=str, default=INPUT_FOLDER)
    parser.add_argument('--output_folder', type=str, default=OUTPUT_FOLDER)
    parser.add_argument('--target_sr', type=int, default=TARGET_SR, help='Target sample rate for resampling')

    args = parser.parse_args()

    device = args.device
    target_sr = args.target_sr

    os.makedirs(args.output_folder, exist_ok=True)

    filelist = get_filelist(args.input_folder)

    print(f"*** Resampling all audio files to {target_sr} Hz ***")
    start_time = time.time()

    with torch.no_grad():
        for file_path in tqdm(filelist):
            wav, sr = torchaudio.load(file_path)
            wav = wav.to(device)

            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
                wav = resampler(wav)

            rel_path = os.path.relpath(file_path, start=args.input_folder)
            output_path = os.path.join(args.output_folder, rel_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.cpu(), target_sr)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
