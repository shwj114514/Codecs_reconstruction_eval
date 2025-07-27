from wrapper import HubertWrapper, SpeachTokenzierWrapper, DAC_16kWrapper, EncodecWrapper, XcodecWrapper,Encodec_uniaudioWrapper
import os
import torch
import argparse
from tqdm import tqdm


def find_all_audio_files(folder_path):
    audio_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(root, file))
    return audio_files


def process_audio_files(audio_files, audio_tokenizer, output_folder, n_quantizers):
    os.makedirs(output_folder, exist_ok=True)
    for audio_file in tqdm(audio_files):
        wav = audio_tokenizer.load(audio_file)
        latent = audio_tokenizer.get_latent(wav, n_quantizers)
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        code_file = os.path.join(output_folder, f"{base_name}.pt")
        torch.save(latent, code_file)


def build_model(model_name, device):
    if model_name == 'Hubert':
        return HubertWrapper(device=device)
    elif model_name == 'SpeachTokenzier':
        return SpeachTokenzierWrapper(device=device,pre_trained_folder="/Codecs/ckpt/SpeechTokenizer")
    elif model_name == 'DAC_16k':
        return DAC_16kWrapper(device=device)
    elif model_name == 'Encodec':
        return EncodecWrapper(device=device)
    elif model_name == 'Xcodec':
        return XcodecWrapper(device=device,pre_trained_folder="/aifs4su/data/zheny/DiT_TTS/tmp/codec_final_0518_20w_ckpts_00400000")
    elif model_name == 'Encodec_uniaudio':
        return Encodec_uniaudioWrapper(device=device,pre_trained_folder ="/aifs4su/data/zheny/fairseq/vae_v2/UniAudio_ori/universal_model")
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio files using Xcodec model")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--audio_folder', type=str,
                        default="/audiolm-abx/data/LibriSpeech",
                        help='Folder containing audio files'
                        )
    parser.add_argument('--model', type=str, default='Encodec',
                        choices=['Hubert', 'SpeachTokenzier', 'DAC_16k', 'Encodec', 'Xcodec', 'Encodec_uniaudio'],
                        help='Model to use for processing audio files'
                        )
    parser.add_argument('--latent_output_folder', type=str,
                        default="/audiolm-abx/features",
                        help='Folder to save the output codes'
                        )
    parser.add_argument('--n_quantizers', type=int, default=12, help='Number of quantizers for encoding')

    args = parser.parse_args()
    audio_tokenizer = build_model(args.model, args.device)

    audio_files = find_all_audio_files(args.audio_folder)
    print(f"code in {args.model}_nq={args.n_quantizers}")
    process_audio_files(audio_files, audio_tokenizer, os.path.join(args.latent_output_folder, f"{args.model}_nq={args.n_quantizers}"),args.n_quantizers)
