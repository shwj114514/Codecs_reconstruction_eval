# Codecs_reconstruction_eval
Codecs_reconstruction_eval is an evaluation toolkit that wraps a broad collection of codec-reconstruction APIs under a single interface, letting you decode audio with one call and instantly compute an extensive set of objective metrics—including 
- PESQ(NB/WB)
- STOI
- Speaker-embedding similarity(SIM)
- Mel-spectrogram loss
- Word-error rate (WER) on LibriSpeech-test-clean
- UMOS
- Usage and entropy

Ready-to-run scripts are provided, and you can define additional metrics in the `metrics`

Supported models include  

- [DAC](https://github.com/descriptinc/descript-audio-codec)  
- [EnCodec](https://github.com/facebookresearch/encodec)  
- [EnCodec of UniAudio](https://github.com/yangdongchao/UniAudio/tree/main/codec)  
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)  
- [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
- [Mimi](https://github.com/kyutai-labs/moshi)  
- [SemantiCodec](https://github.com/haoheliu/SemantiCodec-inference)  
- [HiFiCodec](https://github.com/yangdongchao/AcademiCodec/tree/master/egs/HiFi-Codec-24k-320d)  
- [StableCodec](StableCodec)  
- [FACodec](https://github.com/open-mmlab/Amphion/blob/main/models/codec/ns3_codec/README.md)  
- [BigCodec](https://github.com/Aria-K-Alethia/BigCodec)
- [XCodec](https://github.com/zhenye234/xcodec)  
- [XCodec2](https://github.com/zhenye234/X-Codec-2.0)  

You can define your own model in `wrapper.py`; it needs to inherit from the `AudioTokenizer` class and implement the `load_model`, `get_code`, and `recon_wav` methods.


## environment
```
pip install -r requirements.txt
```
### visqol

[google/visqol: Perceptual Quality Estimator for speech and audio  github.com ](https://github.com/google/visqol?tab=readme-ov-file)

```sh
# visqol
bazel-5.3.2-installer-linux-x86_64.sh

git clone https://github.com/google/visqol.git

bazel build :visqol -c opt
```
The following situations may occur:
```
ImportError: ~/miniconda3/envs/py310/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by ~/miniconda3/envs/py310/lib/python3.10/site-packages/visqol/visqol_lib_py.s)
```
Refer to

[解决 libstdc++.so.6: version ‘GLIBCXX_3.4.30‘ not found 问题_libstdc++.so.6 not found-CSDN博客](https://blog.csdn.net/bohrium/article/details/126546521)

Delete `libstdc++.so.6`
```sh
cd ~/miniconda3/envs/py310/lib
strings libstdc++.so.6 | grep GLIBCXX_3.4.30
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 |grep GLIBCXX_3.4.30

export PATH=$PATH:～/bin
```

## How to use
### test inference
first,you can run `python wrapper.py` to get the bitrate or latent dimension of the codec.
### prepare audio
Many works evaluate speech tokenizers on LibriSpeech/test-clean and we use this dataset as an example. First, download test-clean.tar.gz from https://www.openslr.org/12 and extract or move its contents to `exp_recon/test-clean/`.
```sh
mkdir exp_recon
mv path/to/LibriSpeech/test-clean exp_recon/test-clean_flac
```
and then convert the audio into WAV format.
```sh
python trans_folder_to_wav.py
```


We explicitly store the resampled 16 kHz audio for evaluation.
```sh
python resample_folder.py
```
### reconstruct audio
During evaluation, each audio clip is then resampled to the codec’s sampling rate for reconstruction, and afterward resampled back to 16 kHz for storage and evaluation. 
Run `recon_folder.sh` (or `recon_folder_multi.sh` if you have multiple GPUs) to reconstruct the audio clips in exp_recon/test-clean using your codec.

You will get a folder structure as shown below:
```
exp_recon/
├── DAC_24k_9         # Reconstructed audio using DAC (24kHz) model with 9 RVQ codebooks
├── test-clean        # Original audio clips (original sampling rate)
└── test-clean_16000  # Resampled original audio clips at 16 kHz
└── test-clean_flac
```
### run eval
For pairwise metrics such as PESQ, STOI, and mel distance—where two audio folders must be compared—both folders should have the same sampling rate.
```sh
bash run_pesq_stoi.sh
bash run_mel_stft.sh
```
for other metric,run
```sh
bash run_usage.sh
bash run_entropy.sh
bash run_wer.sh
bash run_umos.sh 
bash run_spk.sh
```

## Acknowledgements
This toolkit reuses code cloned directly from the following projects to simplify setup: 
- [BigCodec](https://github.com/Aria-K-Alethia/BigCodec)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [EnCodec of UniAudio](https://github.com/yangdongchao/UniAudio/tree/main/codec) 
- [FACodec](https://github.com/open-mmlab/Amphion/blob/main/models/codec/ns3_codec/README.md)  
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [XCodec](https://github.com/zhenye234/xcodec)  


