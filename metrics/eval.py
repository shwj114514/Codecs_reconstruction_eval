import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import librosa
import torch.nn.functional as F
import glob
import numpy as np

import torch
from losses import SISDRLoss,MultiScaleSTFTLoss,MelSpectrogramLoss
import os
import argparse
from typing import List
from tqdm import tqdm
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]
from audiotools import AudioSignal

DEVICE = "cuda:0"



# define metric
stft_loss = MultiScaleSTFTLoss().to(DEVICE)  
mel_loss = MelSpectrogramLoss().to(DEVICE)  
sisdr_loss = SISDRLoss().to(DEVICE)  

def visqol(
    estimates: np.ndarray,
    references: np.ndarray,
    mode: str = "audio",
    sr :int =16000
):  # pragma: no cover
    """ViSQOL score.
    Tensor[float]
        ViSQOL score (MOS-LQO)
    """
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)


    visqols = []
    _visqol = api.Measure(
        references.astype(float),
        estimates.astype(float),
    )
    visqols.append(_visqol.moslqo)
    return torch.from_numpy(np.array(visqols))

def safe_pesq(sr, x, y):
    try:
        return pesq(sr, x, y)
    except Exception as e:
        # print(f"Error computing PESQ: {e}")
        return np.nan  # 返回 np.nan 当发生错误
from pystoi import stoi
from pesq import pesq

from audiotools import metrics
def get_metrics(signal_path, recons_path,sr= 16000):
    output = {}
    x_np,_ = librosa.load(signal_path,sr=sr)
    y_np,_ = librosa.load(recons_path,sr=sr)


    # y_np = y_np[:x_np.shape[-1]]
    min_length = min(x_np.shape[-1], y_np.shape[-1])
    x_np = x_np[:min_length]
    y_np = y_np[:min_length]

    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(1).to(DEVICE)     # [1, 1, 48000])
    y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(1).to(DEVICE)  
    # x = torch.from_numpy(x_np).unsqueeze(0)
    # y = torch.from_numpy(y_np).unsqueeze(0)
    k = "16k"
    output.update(
        {
            f"mel-{k}": mel_loss(x, y),
            f"stft-{k}": stft_loss(x, y),
            f"waveform-{k}":  F.l1_loss(x, y),
            f"sisdr-{k}": sisdr_loss(x, y),
            # f"visqol-audio-{k}": visqol(x_np, y_np),
            # f"visqol-speech-{k}": visqol(x_np, y_np, "speech"),
            # rebutal
            f"pesq-{k}":  safe_pesq(16000,x_np, y_np),
            f"stoi-{k}": stoi(x_np, y_np,16000),

        }
    )
    output["path"] = os.path.basename(signal_path)
    return output


# from audiotools import metrics
# def get_metrics(signal_path, recons_path,sr= 16000):
#     output = {}

#     signal = AudioSignal(signal_path)
#     recons = AudioSignal(recons_path)
#     x = signal.clone()
#     y = recons.clone()

#     # y_np = y_np[:x_np.shape[-1]]
#     # x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(1)
#     # y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(1)
#     k = "16k"
#     output.update(
#         {
#             f"mel-{k}": mel_loss(x, y),
#             f"stft-{k}": stft_loss(x, y),
#             f"waveform-{k}":  F.l1_loss(x, y),
#             f"sisdr-{k}": sisdr_loss(x, y),
#             f"visqol-audio-{k}": metrics.quality.visqol(x, y),
#             f"visqol-speech-{k}": metrics.quality.visqol(x, y, "speech"),
#         }
#     )
#     output["path"] = os.path.basename(signal_path)[0]
#     return output



@torch.no_grad()
def evaluate(ori_folder,recon_folder,csv_folder,n_proc):

    os.makedirs(csv_folder,exist_ok=True)
    from utils import get_filelist
    audio_files = get_filelist(ori_folder)
    output = Path(recon_folder)

    def record(future, writer):
        o = future.result()
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        o.pop("path")
        return o

    model_name = recon_folder.split('/')[-2]
    cls_name = recon_folder.split('/')[-1]

    futures = []
    with open(os.path.join(csv_folder ,f"{model_name}_{cls_name}.csv"), "w") as csvfile:
        with ProcessPoolExecutor(n_proc, mp.get_context("spawn")) as pool:
            for i in range(len(audio_files)):
                # print(f"audio_files[i]={audio_files[i]}")
                # print(f"output / audio_files[i].name={output / audio_files[i].name}")
                future = pool.submit(
                    get_metrics, audio_files[i], output / audio_files[i].name,sr= 44100
                )
                futures.append(future)

            keys = list(futures[0].result().keys())
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()

            for future in futures:
                record(future, writer)



    #   # single_infer
    # with open(output / "metrics.csv", "w") as csvfile:
    #     for i in range(len(audio_files)):
    #         print(f"audio_files[i]={audio_files[i]}")
    #         print(f"output / audio_files[i].name={output / audio_files[i].name}")
    #         output = get_metrics(audio_files[i], output / audio_files[i].name, state)
    #         # import pdb;pdb.set_trace()
    #         keys = list(output.keys())
    #         writer = csv.DictWriter(csvfile, fieldnames=keys)
    #         writer.writeheader()
from tqdm import tqdm

# sub_folder = ["speech","events","music"]
sub_folder = ["speech"]
# sub_folder = ["events","music"]

def main(args):
    ori_folder = args.ori_folder
    n_proc = args.n_proc 
    csv_folder = args.csv_folder

    # recon_folders = os.listdir(args.recon_folder)
    # recon_folders = ['Baseline_general_nq=12', 'Xcodec_general_nq=12', 'Xcodec_general_nq=1', 'Baseline_speech_nq=1', 'Baseline_speech_nq=12', 'Baseline_general_nq=1', 'Xcodec_speech_nq=1', 'Xcodec_speech_nq=12']
    recon_folders = [
        # 'Baseline_general_nq=12',
        # 'Xcodec_general_nq=12',
        # 'Xcodec_general_nq=1',
        # 'Baseline_speech_nq=1',
        # 'Baseline_speech_nq=12',
        # 'Baseline_general_nq=1',
        # 'Xcodec_speech_nq=1',
        # 'Xcodec_speech_nq=12'

        # 'SemanticCodec_nq=1',
        # 'SemanticCodec_nq=4',
        # 'HIFICodec_universal_nq=4',
        # 'HIFICodec_nq=4',
        # 'FACodec_nq=6'

        'Baseline_speech_nq=1',
        'Baseline_speech_nq=12',
        'Xcodec_nq=1',
        'Xcodec_nq=12'

    ]

    # recon_folders = ['recon_SoundStream_nq=8']
    for recon_folder in tqdm(recon_folders):

        evaluate(ori_folder,recon_folder,csv_folder,n_proc)




if __name__ == "__main__":



    torch.multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_folder", type=str, default="ori_16000", help="original singing folder")
    parser.add_argument("--csv_folder", type=str, default="csv_metrics", help="generated singing folder")
    parser.add_argument("--recon_folder", default="recon",type=str)
    parser.add_argument("--n_proc", type=int, default=40)

    args = parser.parse_args()
    main(args=args)