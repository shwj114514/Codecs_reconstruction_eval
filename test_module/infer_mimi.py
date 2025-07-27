import torchaudio
from transformers import MimiModel, AutoFeatureExtractor

audio_path = "LJ001-0001.wav"
model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
audio_array, sr = torchaudio.load(audio_path)
if audio_array.shape[0] > 1:
    audio_array = audio_array.mean(dim=0, keepdim=True)
if sr != feature_extractor.sampling_rate:
    resampler = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)
    audio_array = resampler(audio_array)
    sr = feature_extractor.sampling_rate
audio_array = audio_array.squeeze(0).numpy()
inputs = feature_extractor(raw_audio=audio_array, sampling_rate=sr, return_tensors="pt")
encoder_outputs = model.encode(inputs["input_values"])
import pdb;pdb.set_trace()
audio_values = model.decode(encoder_outputs.audio_codes)[0]
audio_values = model(inputs["input_values"]).audio_values
print(audio_values.shape)