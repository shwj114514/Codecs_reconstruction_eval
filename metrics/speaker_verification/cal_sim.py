import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from tqdm import tqdm
import sys
sys.path.append('speaker_verification')
from  verification import init_model
import torch.nn.functional as F
import torchaudio

from torchaudio.transforms import Resample, Vad
model_spk = init_model('wavlm_large','/models_fd_ckpt/wavlm_large_finetune.pth')

model_spk = model_spk.cuda()

model_spk.eval()




def process_audio(file_path, start_time=None, duration=None,trim_silence=False):
    waveform, sample_rate = torchaudio.load(file_path)


    if trim_silence:
        vad = Vad(sample_rate=sample_rate, trigger_level=20)  # 设定静音检测的dB阈值为-20dB
        waveform = vad(waveform)

    # 如果指定了开始时间
    if start_time is not None:
        start_sample = int(start_time * sample_rate)
        if duration is not None:
            # 如果同时指定了持续时间
            end_sample = start_sample + int(duration * sample_rate)
            waveform = waveform[:, start_sample:end_sample]
        else:
            # 如果没有指定持续时间，则读取从开始时间到音频末尾的部分
            waveform = waveform[:, start_sample:]
    elif duration is not None:
        # 如果只指定了持续时间，假设从音频开头开始读取固定持续时间
        end_sample = int(duration * sample_rate)
        waveform = waveform[:, :end_sample]

    # 重采样到指定的采样率（如果需要）
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    return waveform.cuda()

# 计算所有文件对之间的相似度
similarity_scores = []
index=0
# for file1 in folder:

filename1='/valle_gt_pre3s/'

filename2='/valle_xcodec_hubert_baseline_nar_final-epoch-200-continue-8/'
# for i in tqdm(range(123)):
for i in tqdm(range(1229)):

    numbers=index+1

    file1=filename1+str(i)+'.wav'


    embeddings1 = model_spk(process_audio(file1  ))


    file2=filename2+str(i)+'.wav'


    embeddings2 = model_spk(process_audio(file2 ,start_time=3  ))

    # embeddings2 = model_spk(process_audio(file2  ))

    sim = F.cosine_similarity(embeddings1, embeddings2)

    similarity_scores.append(sim.item())
    index = index+1

    average_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f'Average similarity: {average_similarity}')



# one token continue 上限 0.34
# eight token continue 上限 0.58 (xcodec)

# one token 重建 上限 0.52


