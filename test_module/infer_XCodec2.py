import torch
import soundfile as sf
from transformers import AutoConfig


from xcodec2.modeling_xcodec2 import XCodec2Model

from pathlib import Path
audio_path = "LJ001-0001.wav"
audio_path = "2001000001.wav"

save_path = f"XCodec2_{Path(audio_path).stem}.wav"


model_path = "HKUST-Audio/xcodec2"  
device = "cuda:5"


model = XCodec2Model.from_pretrained(model_path)
model.eval().to(device)  


wav, sr = sf.read(audio_path)   
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)


with torch.no_grad():
   # Only 16khz speech
   # Only supports single input. For batch inference, please refer to the link below.
    vq_code = model.encode_code(input_waveform=wav_tensor)
    print("Code:", vq_code )  

    recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')


sf.write(save_path, recon_wav[0, 0, :].numpy(), sr)
print("Done! Check reconstructed.wav")
