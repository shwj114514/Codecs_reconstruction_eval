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
first,you can run `python wrapper.py` to get the bitrate or latent dimension of the codec.

During evaluation, each audio clip is then resampled to the codec’s sampling rate for reconstruction, and afterward resampled back to 16 kHz for storage and evaluation.
```sh
mkdir exp_recon
mv path/to/LibriSpeech/test-clean exp_recon/test-clean
python resample_folder.py
```

then reconstruct the audio of exp_recon/test-clean using your Codec

```sh
python recon_folder.py
```

or
```sh
bash recon_folder.sh
```

For pairwise metrics such as PESQ, STOI, and mel distance—where two audio folders must be compared—both folders should have the same sampling rate.

```sh
run_pesq_stoi.sh
run_wer.sh
```


## Acknowledgements
This toolkit reuses code cloned directly from the following projects to simplify setup: 
- [BigCodec](https://github.com/Aria-K-Alethia/BigCodec)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [EnCodec of UniAudio](https://github.com/yangdongchao/UniAudio/tree/main/codec) 
- [FACodec](https://github.com/open-mmlab/Amphion/blob/main/models/codec/ns3_codec/README.md)  
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer)
- [XCodec](https://github.com/zhenye234/xcodec)  


