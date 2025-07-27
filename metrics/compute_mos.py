import os
import torch
import librosa
import ssl
import argparse
from tqdm import tqdm
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置SSL上下文
ssl._create_default_https_context = ssl._create_unverified_context


def calculate_mos(folder_path):
    # 加载评分模型
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    total_score = 0
    file_count = 0
    # 遍历文件夹及其所有子文件夹
    for filename in tqdm(os.listdir(folder_path)):  # 使用tqdm包装遍历操作
        if filename.endswith((".wav", ".flac")):  # 检查文件扩展名
            file_path = os.path.join(folder_path, filename)

            # 加载音频文件
            wave, sr = sf.read(file_path)

            # 将波形数据转换为适合模型的格式并移动到GPU上
            wave_tensor = torch.from_numpy(wave).unsqueeze(0).float().to(device)

            # 预测评分
            score = predictor(wave_tensor, sr)

            # 累加评分和文件数
            total_score += score.item()
            file_count += 1

    return total_score, file_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute MOS measure")

    parser.add_argument(
        '-f', '--folder_path', required=True, help="Folder containing the audio files.")

    args = parser.parse_args()

    total_score, file_count = calculate_mos(args.folder_path)

    # 打印总评分和处理的文件数量
    print(f"Total score: {total_score}")
    print(f"Total number of files processed: {file_count}")
    print(f"========={args.folder_path}===============")
    # 计算平均评分
    if file_count > 0:
        average_score = total_score / file_count
        print(f"Average Score: {average_score} ")
    else:
        print("No WAV files found in the folder.")
