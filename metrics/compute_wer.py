import os
import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import jiwer
import torchaudio
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer  # Import the normalizer
normalizer = EnglishTextNormalizer()
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

data = []
# 打开文本文件
with open('combined.txt', 'r') as file:
    lines = file.readlines()

data = {}

for line in lines:
    line = line.strip()  # 移除行尾的换行符和空格
    parts = line.split(' ', 1)  # 以第一个空格为分隔符，将句子分为文件名和句子内容两部分
    
    data[parts[0]] = parts[1]  # 使用文件名作为键，句子内容作为值)

from pathlib import Path
import typing as tp
import os


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

TARGET_SR = 16000

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute STOI measure")
    parser.add_argument(
        '-f', '--folder_path', required=True, help="Folder containing the audio files.")

    args = parser.parse_args()

    filelist = get_filelist(folder=args.folder_path)
    DEVICE = "cuda:0"

    model = model.to(DEVICE)  

    print(processor.feature_extractor.sampling_rate)

    c=1 
    trans_gt =[]
    trans_from_audio=[]

    for file_path in tqdm(filelist):
        file_stem = Path(file_path).stem
        transcription_gt = data[file_stem]
        transcription_gt = normalizer(transcription_gt)
        trans_gt.append(transcription_gt)
        wav, sr = torchaudio.load(file_path)
        wav = wav.to(DEVICE)

        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR).to(DEVICE)
            wav = resampler(wav)
        
        wav = wav.squeeze(0)
        
        # must sample to 16k
        input_values = processor(wav, return_tensors="pt",sampling_rate = TARGET_SR).input_values  # Batch size 1

        logits = model(input_values.to(DEVICE)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        transcription = normalizer(transcription)
        trans_from_audio.append(transcription)

    wer = jiwer.wer( trans_gt,trans_from_audio    )
    print(f"WER: {wer * 100:.2f} %")
    #gt 2.08% re-syn   2.11 %

# golden WER: 1.96 %
# 24k WER 1.96
# 24k音频不采样 WER: 4.78 %