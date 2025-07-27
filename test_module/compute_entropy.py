import argparse

import audiotools as at
import numpy as np
import torch
import tqdm

import dac
import sys
sys.path.append("../")
from wrapper import build_model

def cal_entropy(
    folder: str,
    model_path : str = "16khz",
    n_samples: int = 1024,
    codebook_size = 1024,
    device: str = "cuda",
):

    files = at.util.find_audio(folder)[:n_samples]

    signals = [
        at.AudioSignal.salient_excerpt(f, loudness_cutoff=-20, duration=1.0)
        for f in files
    ]

    with torch.no_grad():
        model_path = dac.utils.download(model_type="24khz")
        model = dac.DAC.load(model_path)
        model = model.to(device)
        model.eval()

        codes = []
        for x in tqdm.tqdm(signals):
            x = x.to(model.device)
            # [B, 32, 50]
            z, codes_, latents, commitment_loss, codebook_loss = model.encode(x.audio_data, x.sample_rate)
            codes.append(codes_)
            import pdb;pdb.set_trace()

        # [1, 32, 51200] 1024*50  [1,seq_len,n_q*n_sample]
        codes = torch.cat(codes, dim=-1)
        entropy = []

        import pdb;pdb.set_trace()

        for i in range(codes.shape[1]):
            codes_ = codes[0, i, :]
            counts = torch.bincount(codes_)
            counts = (counts / counts.sum()).clamp(1e-10)
            entropy.append(-(counts * counts.log()).sum().item() * np.log2(np.e))

        pct = sum(entropy) / (np.log2(codebook_size) * len(entropy))
        print(f"Entropy for each codebook: {entropy}")
        import pdb;pdb.set_trace()
        print(f"Effective percentage: {pct * 100}%")

        return pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

    parser.add_argument('--model', type=str, default='SpeachTokenzier',help='Model to use for processing audio files')
    parser.add_argument('--n_samples', type=int, default=1024)
    parser.add_argument('--n_quantizers', type=int, default=2, help='Number of quantizers for encoding')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')

    args = parser.parse_args()
    TMP_FOLDER = "/exp_recon/tmp"
    pct = cal_entropy(TMP_FOLDER)

    audio_tokenizer = build_model(args.model, args.device)
    sr = audio_tokenizer.sample_rate

