import argparse

import audiotools as at
import numpy as np
import torch
import torchaudio
import tqdm

import dac

from wrapper import build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PESQ measure.")


    parser.add_argument(
        '-f', '--folder_path', required=True, help="Folder containing the audio files.")
    parser.add_argument('--input_folder', type=str,
                        default="exp_recon/test-clean",
                        help='Folder containing audio files'
                        )
    parser.add_argument('--model', type=str, default='Encodec',help='Model to use for processing audio files')
    parser.add_argument('--n_samples', type=int, default=1024)
    parser.add_argument('--n_quantizers', type=int, default=1, help='Number of quantizers for encoding')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')

    args = parser.parse_args()


    model = build_model(args.model, args.device)
    model_sr = model.sample_rate
    codebook_size = model.codebook_size

    n_q = args.n_quantizers
    folder_path = args.folder_path
    n_samples = args.n_samples

    files = at.util.find_audio(folder_path)[:n_samples]

    signals = [
        at.AudioSignal.salient_excerpt(f, loudness_cutoff=-20, duration=1.0)
        for f in files
    ]

    with torch.no_grad():


        codes = []
        for x in tqdm.tqdm(signals):
            x = x.to(model.device)
            wav = x.audio_data.squeeze(0)
            sr = x.sample_rate

            if sr != model_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
                resampler = resampler.to(wav.device)
                wav = resampler(wav)
                sr = model_sr
            codes_ = model.get_code(wav = wav, n_q = n_q)
            # [B, n_q, 50]
            codes.append(codes_)

        # [1,n_q,50*n_sample]
        codes = torch.cat(codes, dim=-1)
        entropy = []

        for i in range(codes.shape[1]):
            codes_ = codes[0, i, :]
            counts = torch.bincount(codes_)
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))

        pct = sum(entropy) / (np.ceil(np.log2(codebook_size)) * len(entropy))
        print(f"========={args.model}===============")

        print(f"Entropy for each codebook: {entropy}")
        print(f"Effective percentage: {pct * 100}%")

