import argparse
import glob
import os

import scipy.signal as signal
from pesq import pesq
from scipy.io import wavfile
from tqdm import tqdm

from utils import get_filelist

import soundfile as sf

def cal_pesq(ref_dir, deg_dir):
    # input_files = glob.glob(f"{deg_dir}/*.wav")
    filelist = get_filelist(deg_dir)
    nb_pesq_scores = 0.0
    wb_pesq_scores = 0.0
    cnt = 0
    for deg_wav in tqdm(filelist):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        # ref_rate, ref = wavfile.read(ref_wav)
        # deg_rate, deg = wavfile.read(deg_wav)
        
        ref, ref_rate = sf.read(ref_wav)
        deg, deg_rate = sf.read(deg_wav)


        if ref_rate != 16000:
            ref = signal.resample(ref, 16000)
        if deg_rate != 16000:
            deg = signal.resample(deg, 16000)

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        try:
            nb_pesq_scores += pesq(16000, ref, deg, 'nb')
            wb_pesq_scores += pesq(16000, ref, deg, 'wb')
            cnt +=1
        except Exception as e:
            print(f"e = {e} in deg_wav = {deg_wav}")
            
    return nb_pesq_scores / cnt, wb_pesq_scores / cnt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

    parser.add_argument(
        '-r', '--ref_dir', required=True, help="Reference wave folder.")
    parser.add_argument(
        '-d', '--deg_dir', required=True, help="Degraded wave folder.")

    args = parser.parse_args()

    nb_score, wb_score = cal_pesq(args.ref_dir, args.deg_dir)
    print(f"NB PESQ: {nb_score}")
    print(f"WB PESQ: {wb_score}")
