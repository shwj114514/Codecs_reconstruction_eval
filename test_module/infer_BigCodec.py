import os
import librosa

import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join, exists

from argparse import ArgumentParser
from time import time

import torch
import torch.nn.functional as F

import sys
sys.path.append("../")

from BigCodec.vq.codec_encoder import CodecEncoder
from BigCodec.vq.codec_decoder import CodecDecoder



from pathlib import Path
audio_path = "LJ001-0001.wav"
audio_path = "2001000001.wav"

save_path = f"BigCodec_{Path(audio_path).stem}.wav"
CKPT_PATH = "/home/jiahelei/DATA/pretrained_models/BigCodec/bigcodec.pt"


if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('--input-dir', type=str, default='.')
    # parser.add_argument('--ckpt', type=str, default='bigcodec.pt')
    # parser.add_argument('--output-dir', required=True, type=str, default='outputs')



    sr = 16000

    print(f'Load codec ckpt from {CKPT_PATH}')
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    encoder = CodecEncoder()
    encoder.load_state_dict(ckpt['CodecEnc'])
    encoder = encoder.eval()
    decoder = CodecDecoder()
    decoder.load_state_dict(ckpt['generator'])
    decoder = decoder.eval()



    wav = librosa.load(audio_path, sr=sr)[0] 
    wav = torch.from_numpy(wav).unsqueeze(0)
    wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
    with torch.no_grad():
        vq_emb = encoder(wav.unsqueeze(1))
        vq_post_emb, vq_code, _ = decoder(vq_emb, vq=True)
        recon = decoder(vq_post_emb, vq=False).squeeze().detach().cpu().numpy()
    sf.write(save_path, recon, sr)
et = time()

    # st = time()
    # for wav_path in tqdm(wav_paths):
    #     target_wav_path = join(wav_dir, basename(wav_path))
    #     wav = librosa.load(wav_path, sr=sr)[0] 
    #     wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    #     wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
    #     with torch.no_grad():
    #         vq_emb = encoder(wav.unsqueeze(1))
    #         vq_post_emb, vq_code, _ = decoder(vq_emb, vq=True)
    #         recon = decoder(vq_post_emb, vq=False).squeeze().detach().cpu().numpy()
    #     sf.write(target_wav_path, recon, sr)
    # et = time()
    # print(f'Inference ends, time: {(et-st)/60:.2f} mins')
