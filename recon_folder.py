import argparse

from tqdm import tqdm

from pathlib import Path
import typing as tp
import os

import torch
import torchaudio
from wrapper import build_model

def get_filelist(
        folder: tp.Union[str, os.PathLike],
        extensions: tp.Optional[tp.List[str]] = None
) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)

    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    filelist = [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]
    return filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files using Xcodec model")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--input_folder', type=str,
                        default="exp_recon/test-clean",
                        help='Folder containing audio files'
                        )
    parser.add_argument('--model', type=str, default='Encodec',help='Model to use for processing audio files')
    parser.add_argument('--output_folder', type=str,default="exp_recon/DUMMY")
    parser.add_argument('--n_quantizers', type=int, default=12, help='Number of quantizers for encoding')
    parser.add_argument('--target_sr', type=int, default=16000)



    args = parser.parse_args()
    output_folder = args.output_folder
    input_folder = args.input_folder
    n_q = args.n_quantizers
    target_sr = args.target_sr

    os.makedirs(output_folder, exist_ok=True)

    audio_tokenizer = build_model(args.model, args.device)
    sr = audio_tokenizer.sample_rate

    print(f"***  model =  {args.model}  nq={args.n_quantizers} sample_rate = {audio_tokenizer.sample_rate}  ****")
    filelist = get_filelist(input_folder)
    import time
    start_time = time.time()
    with torch.no_grad():
        for file_path in tqdm(filelist):
            wav = audio_tokenizer.load(file_path)
            recon_wav = audio_tokenizer.recon_wav(wav,n_q = n_q)

            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                resampler = resampler.to(recon_wav.device)
                recon_wav = resampler(recon_wav)

            rel_path = os.path.relpath(file_path, start=input_folder)
            output_path = os.path.join(output_folder, rel_path)

            torchaudio.save(output_path, recon_wav.cpu(), target_sr)
    end_time = time.time()
    print(f"time = { end_time - start_time}")




