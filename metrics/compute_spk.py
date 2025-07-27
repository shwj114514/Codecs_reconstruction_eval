import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from torchaudio.transforms import Resample, Vad

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

current_script_dir = os.path.dirname(os.path.realpath(__file__))
sv_path = os.path.join(current_script_dir, 'speaker_verification')
sys.path.append(sv_path)
from speaker_verification.verification import init_model

# 初始化模型

model_spk = init_model('wavlm_large')

model_spk = model_spk.cuda()
model_spk.eval()

def process_audio(file_path, start_time=None, duration=None, trim_silence=False,sr= 16000):
    waveform, sample_rate = torchaudio.load(file_path)

    if trim_silence:
        vad = Vad(sample_rate=sample_rate, trigger_level=20)
        waveform = vad(waveform)

    if start_time is not None:
        start_sample = int(start_time * sample_rate)
        if duration is not None:
            end_sample = start_sample + int(duration * sample_rate)
            waveform = waveform[:, start_sample:end_sample]
        else:
            waveform = waveform[:, start_sample:]
    elif duration is not None:
        end_sample = int(duration * sample_rate)
        waveform = waveform[:, :end_sample]

    if sample_rate != sr:
        resampler = Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    
    return waveform.cuda()

def compute_similarity(ref_dir, gen_dir):
    similarity_scores = []

    ref_files = [f for f in os.listdir(ref_dir) if f.endswith('.wav')]
    
    # from utils import get_filelist
    # ref_files = get_filelist(ref_dir)
    for ref_file in tqdm(ref_files):
        file1 = os.path.join(ref_dir, ref_file)
        file2 = os.path.join(gen_dir, ref_file)
        
        if not os.path.exists(file2):
            continue
        
        embeddings1 = model_spk(process_audio(file1))
        embeddings2 = model_spk(process_audio(file2))

        sim = F.cosine_similarity(embeddings1, embeddings2)
        similarity_scores.append(sim.item())

    average_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    print(f"================{gen_dir}============")
    print(f'Average similarity: {average_similarity}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute similarity scores between audio files")
    parser.add_argument('-r', '--ref_dir', required=True, help="Reference folder containing audio files")
    parser.add_argument('-g', '--gen_dir', required=True, help="Generated folder containing audio files")

    args = parser.parse_args()

    compute_similarity(args.ref_dir, args.gen_dir)
