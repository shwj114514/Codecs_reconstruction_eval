import torch
from transformers import AutoModel, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
import os
import torchaudio
import torch.nn as nn
from omegaconf import OmegaConf
from einops import rearrange
import torch.nn.functional as F
import math

HF_TOKEN = ""

class AudioTokenizer():
    """A Wrapper class for semantic and accoustic tokenizer
    """

    def __init__(self, pre_trained_folder: str = None, sample_rate: int = 16000, device: str = None) -> None:
        super().__init__()
        self.pre_trained_folder = pre_trained_folder
        self.sample_rate = sample_rate
        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.load_model()
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def load_model(self):
        raise NotImplementedError

    def load(self, audio_file: str):
        wav, original_sr = torchaudio.load(audio_file)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
            wav = resampler(wav)
        return wav.to(self.device)

    def load_model(self):
        raise NotImplementedError

    def get_code(self, wav: torch.Tensor, n_q: int = None) -> torch.Tensor:
        # [B,n_q, seq]
        raise NotImplementedError

    def get_latent(self, wav: torch.Tensor, n_q: int = None) -> torch.Tensor:
        # latent:[Sequence_size , Feature_dimension]
        raise NotImplementedError

    def get_embedding(self, wav: torch.Tensor, n_q: int = None) -> torch.Tensor:
        latent = self.get_latent(wav=wav, n_q=n_q)
        if len(latent.shape) > 2:
            latent = latent.unsqueeze(0)
        assert len(latent.shape) == 2  # [T,dim]
        return latent.mean(dim=0)  # dim

    def recon_wav(self, wav: torch.Tensor, n_q: int = None) -> torch.Tensor:
        raise NotImplementedError  # [1,N]

    def nq_to_bw(self, n_q=8, down_rate=320, codes_dim=1024):
        return n_q * math.log2(codes_dim) * self.sample_rate / down_rate / 1000

    def sum_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def eval(self):
        # for evaluate project
        pass


class HubertWrapper(AudioTokenizer):
    def __init__(self, pre_trained_folder="facebook/hubert-base-ls960", sample_rate=16000, device: str = None,
            hidden_size=768):
        self.hidden_size = hidden_size
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0").from_pretrained(
            'facebook/w2v-bert-2.0')
        super().__init__(pre_trained_folder, sample_rate, device)

    def load_model(self):
        model = AutoModel.from_pretrained(self.pre_trained_folder)
        return model

    @torch.no_grad()
    def get_latent(self, wav, n_q=None):
        # wav:[1,N]
        wav = self.processor(wav.cpu(), sampling_rate=self.sample_rate, return_tensors="pt")['input_features'].to(
            self.device)
        latent = self.model(wav).last_hidden_state.squeeze()
        return latent


class Encodec_uniaudioWrapper(AudioTokenizer):
    def __init__(self, pre_trained_folder, sample_rate=16000, device: str = None):
        self.hidden_size = 256
        super().__init__(pre_trained_folder, sample_rate, device)

    def load_model(self):
        # universal_model
        from Encodec_uniaudio.soundstream import SoundStream
        model_path = os.path.join(self.pre_trained_folder, "ckpt_01455000.pth")
        config_path = os.path.join(self.pre_trained_folder, "config.yaml")
        parameter_dict = torch.load(model_path, map_location='cpu')
        config = OmegaConf.load(config_path)
        # model = eval(config.generator.name)(**config.generator.config)
        model = SoundStream(**config.generator.config)
        model.load_state_dict(parameter_dict['codec_model'])
        return model

    def get_latent(self, wav: torch.Tensor, n_q: int = 8) -> torch.Tensor:
        e = self.model.encoder(wav.unsqueeze(0))
        bw = self.nq_to_bw(n_q=n_q, down_rate=320)

        codes = self.model.quantizer.encode(e, self.model.frame_rate, bw)
        quantized = self.model.quantizer.decode(codes)
        latent = quantized.squeeze().transpose(0, 1)
        return latent  # [256,T]

    def recon_wav(self, wav: torch.Tensor, n_q: int = 8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        bw = self.nq_to_bw(n_q=n_q, down_rate=320)
        codes = self.model.encode(wav, bw)
        recon_wav = self.model.decode(codes).squeeze(0)  # [1, 1, 16000]
        return recon_wav


class SpeachTokenzierWrapper(AudioTokenizer):
    def __init__(self, sample_rate=16000, device: str = None):
        self.hidden_size = 768
        self.codebook_size = 1024
        self.device = device

        super().__init__( sample_rate = sample_rate, device = device)

    def load_model(self):
        from speechtokenizer import SpeechTokenizer
        # config_path = os.path.join(self.pre_trained_folder, "config.json")
        # ckpt_path = os.path.join(self.pre_trained_folder, "SpeechTokenizer.pt")

        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id="fnlp/SpeechTokenizer",
                                    filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt")

        # 下载配置文件
        config_path = hf_hub_download(repo_id="fnlp/SpeechTokenizer",
                                    filename="speechtokenizer_hubert_avg/config.json")

        model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        return model

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        e = self.model.encoder(wav)  # [1, 1024, T]
        quantized, codes, commit_loss, quantized_list = self.model.quantizer(e, n_q=n_q, layers=None)
        feature = rearrange(quantized, 'b d t -> b t d')
        feature = self.model.transform(feature)  # [1, T, 768])
        latent = feature.squeeze()
        return latent

    def get_code(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        # all_codes = self.model.encode(wav)  # codes: (n_q, B, T)
        # codes = all_codes[:n_q, :, :]  # Contain content info, can be considered as semantic tokens
        codes = self.model.encode(wav, n_q=n_q)
        codes = codes.transpose(0, 1)
        return codes

    def recon_wav(self, wav, n_q=None):
        codes = self.get_code(wav, n_q=n_q)
        # [1, n_q, seq_len]  [n_q, 1, seq_len]
        codes = codes.transpose(0, 1)
        recon_wav = self.model.decode(codes).squeeze(0)  # [1, 1, 16000]
        return recon_wav


class EncodecWrapper(AudioTokenizer):
    def __init__(self,  sample_rate=24000, device: str = None):
        self.hidden_size = 128
        self.codebook_size = 1024
        self.device = device

        super().__init__( sample_rate = sample_rate, device = device)

    def load_model(self):
        from encodec import EncodecModel
        from encodec.utils import convert_audio
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6)
        return model

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        frames = self.model.encode(wav)  # frames[0][0].shape =
        all_codes = frames[0][0]  # [1,8,T]
        codes = all_codes[:, :n_q, :]
        codes = codes.transpose(0, 1)  # [8, 1, 375]
        latent = self.model.quantizer.decode(codes).squeeze(0).transpose(0, 1)
        return latent

    def get_code(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        frames = self.model.encode(wav)  # frames[0][0].shape =
        all_codes = frames[0][0]  # [1,8,T]
        codes = all_codes[:, :n_q, :]
        return codes

    def recon_wav(self, wav, n_q=None):
        if n_q == 8:
            wav = wav.unsqueeze(0)
            frames = self.model.encode(wav)
            recon_wav = self.model.decode(frames).squeeze(0)
            return recon_wav
        codes = self.get_code(wav, n_q)
        codes = codes.transpose(0, 1)  # [8, 1, 375]
        emb = self.model.quantizer.decode(codes)
        recon_wav = self.model.decoder(emb).squeeze(0)

        return recon_wav


class DAC_16kWrapper(AudioTokenizer):
    def __init__(self, sample_rate=16000, device: str = None):
        self.hidden_size = 1024
        self.codebook_size = 1024
        super().__init__( sample_rate = sample_rate, device = device)

    def load_model(self):
        import dac
        model_path = dac.utils.download(model_type="16khz")
        model = dac.DAC.load(model_path)
        return model

    def get_latent(self, wav: torch.Tensor, n_q=12) -> torch.Tensor:
        from audiotools import AudioSignal
        signal = AudioSignal(wav, sample_rate=self.sample_rate)
        signal.to(self.model.device)

        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x, n_quantizers=n_q)
        latent = z.squeeze().transpose(0, 1)

        return latent
    
    def get_code(self, wav: torch.Tensor, n_q=12) -> torch.Tensor:
        from audiotools import AudioSignal
        signal = AudioSignal(wav, sample_rate=self.sample_rate)
        signal.to(self.model.device)

        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x, n_quantizers=n_q)

        return codes

    def recon_wav(self, wav, n_q=None):
        from audiotools import AudioSignal
        signal = AudioSignal(wav, sample_rate=self.sample_rate)
        signal.to(self.model.device)

        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x, n_quantizers=n_q)
        recon_wav = self.model.decode(z).detach().squeeze(0)
        return recon_wav

class DAC_24kWrapper(DAC_16kWrapper):
    def __init__(self, sample_rate=24000, device: str = None):
        self.hidden_size = 1024
        self.codebook_size = 1024

        AudioTokenizer.__init__(self,  sample_rate = sample_rate, device = device)
    def load_model(self):
        import dac
        model_path = dac.utils.download(model_type="24khz")
        model = dac.DAC.load(model_path)
        return model

class DAC_44kWrapper(DAC_16kWrapper):
    def __init__(self, sample_rate=44100, device: str = None):
        self.hidden_size = 1024
        self.codebook_size = 1024

        AudioTokenizer.__init__(self,  sample_rate = sample_rate, device = device)
    def load_model(self):
        import dac
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)
        return model

class SemantiCodecWrapper(AudioTokenizer):
    def __init__(self,bps:int =700, sample_rate=16000, device: str = None):
        self.hidden_size = 768
        self.sample_rate = sample_rate
        self.device = device
        self.bps = bps

        from semanticodec import SemantiCodec
        self.model = self.load_model()
        self.model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def load_model(self):
        from semanticodec import SemantiCodec
        if self.bps == 1400:
            semanticodec = SemantiCodec(token_rate=100, semantic_vocab_size=32768)
            # audimae semanticodec: 8192  13
            self.codebook_size = 32768
        elif self.bps == 700:
            semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=32768)
            self.codebook_size = 32768
        elif self.bps == 680:
            # 14 * 25.2 + 13 *25.2 = 680.4
            semanticodec = SemantiCodec(token_rate=50, semantic_vocab_size=16384) # 0.68 kbps
            self.codebook_size = 16384

        else:
            import pdb;pdb.set_trace()

        return semanticodec


    def encode_from_tensor(self, wav: torch.Tensor):
        sr = 16000
        SAMPLE_RATE = 16000
        SEGMENT_DURATION = 10.24
        MEL_TARGET_LENGTH = 1024
        AUDIOMAE_PATCH_DURATION = 0.16
        SEGMENT_OVERLAP_RATIO = 0.0625
        from semanticodec.utils import extract_kaldi_fbank_feature

        original_duration = wav.shape[1] / sr
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
                AUDIOMAE_PATCH_DURATION - original_duration % AUDIOMAE_PATCH_DURATION
        )
        # Calculate the token length in theory
        target_token_len = (
                8 * original_duration / AUDIOMAE_PATCH_DURATION / self.model.stack_factor_K
        )
        segment_sample_length = int(SAMPLE_RATE * SEGMENT_DURATION)
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations
        if wav.shape[1] % segment_sample_length < segment_sample_length:
            waveform = torch.cat(
                [
                    wav,
                    torch.zeros(
                        1,
                        int(
                            segment_sample_length
                            - wav.shape[1] % segment_sample_length
                        ),device=self.device
                    ),
                ],
                dim=1,
            )

        mel_target_length = MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        # Calculate the mel spectrogram
        mel = extract_kaldi_fbank_feature(
            wav, sr, target_length=mel_target_length
        )["ta_kaldi_fbank"].unsqueeze(0)
        mel = mel.squeeze(1)
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0

        tokens = self.model.encoder(mel.to(self.device))
        tokens = tokens[:, : math.ceil(target_token_len), :]

        return tokens    

    def get_code(self, wav: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_from_tensor(wav)
        code = tokens.transpose(1,2)
        import pdb;pdb.set_trace()
        return code
    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        latent = self.model.get_latent(wav)
        return latent  # [256,T]

    def recon_wav(self, wav: torch.Tensor, n_q=None):
        tokens = self.encode_from_tensor(wav)
        recon_wav = self.model.decode(tokens)
        return torch.from_numpy(recon_wav).squeeze(1)

    def sum_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class XcodecWrapper(AudioTokenizer):
    def __init__(self, ckpt_path = None, config_path = None, sample_rate=16000, device: str = None):
        from huggingface_hub import hf_hub_download
        if config_path == None:
            config_path = hf_hub_download(
                repo_id="ZhenYe234/xcodec",
                filename="config_wavlm.yaml",
            )
        if ckpt_path == None:
            ckpt_path = hf_hub_download(
                repo_id="ZhenYe234/xcodec",
                filename="xcodec_speech_wavlm_more_data.pth",
            )

        
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.codebook_size = 1024
        self.hidden_size = 768
        super().__init__(sample_rate=sample_rate, device=device)

    def load_model(self):
        from xcodec.models.soundstream_semantic import SoundStream
        parameter_dict = torch.load(self.ckpt_path)
        config = OmegaConf.load(self.config_path)
        soundstream = eval(config.generator.name)(**config.generator.config)
        parameter_dict = torch.load(self.ckpt_path)
        soundstream.load_state_dict(parameter_dict )  # Load model

        return soundstream
    def get_code(self, wav: torch.Tensor, n_q: int = None) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        bw = self.nq_to_bw(n_q=n_q, down_rate=320)
        codes = self.model.encode(wav, bw)
        codes = codes.transpose(0,1)
        return codes

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        bw = self.nq_to_bw(n_q=n_q, down_rate=320)
        latent = self.model.get_latent(wav, target_bw=bw)
        return latent  # [256,T]

    def recon_wav(self, wav, n_q=None):
        wav = wav.unsqueeze(0)
        bw = self.nq_to_bw(n_q=n_q, down_rate=320)
        codes = self.model.encode(wav, bw)
        recon_wav = self.model.decode(codes).squeeze(0)  # [1, 1, 16000]

        return recon_wav





class FACodecWrapper(AudioTokenizer):
    def __init__(self, sample_rate=16000, device: str = None):
        self.sample_rate = sample_rate

        self.device = device

        # load_model
        from FACodec import FACodecEncoder, FACodecDecoder
        from huggingface_hub import hf_hub_download
        self.hidden_size = 256
        fa_encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
        fa_encoder.load_state_dict(torch.load(encoder_ckpt))

        fa_decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")
        fa_decoder.load_state_dict(torch.load(decoder_ckpt))

        fa_encoder.eval()
        fa_decoder.eval()
        for param in fa_encoder.parameters():
            param.requires_grad = False
        for param in fa_decoder.parameters():
            param.requires_grad = False
        # self.model = nn.Sequential(
        #     fa_encoder,
        #     fa_decoder
        # )
        self.encoder = fa_encoder.to(device)
        self.decoder = fa_decoder.to(device)

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        enc_out = self.encoder(wav)
        # quantize
        vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(enc_out, eval_vq=False, vq=True)

        latent = quantized[1].squeeze(0).transpose(0, 1)
        # latent = quantized[1].squeeze(0)
        return latent  # [T,256]

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        wav = wav.unsqueeze(0)
        enc_out = self.encoder(wav)
        # quantize
        vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(enc_out, eval_vq=False, vq=True)

        if n_q == 2:
            latent = quantized[1].squeeze(0).transpose(0, 1)
        # latent = quantized[1].squeeze(0)
        if n_q == 6:
            latent = vq_post_emb.squeeze(0).transpose(0, 1)

        else:
            import pdb;
            pdb.set_trace()

        return latent  # [T,256]

    def recon_wav(self, wav: torch.Tensor, n_q=None):
        wav = wav.unsqueeze(0)
        # encode
        enc_out = self.encoder(wav)
        # quantize
        vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(enc_out, eval_vq=False, vq=True)
        # decode (recommand)
        recon_wav = self.decoder.inference(vq_post_emb, spk_embs).squeeze(0)

        return recon_wav

    def sum_parameters(self):
        return sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters())


class HIFICodecWrapper(AudioTokenizer):
    def __init__(self, ckpt_path: str, config_path='./hificodec/config_16k_320d.json', sample_rate=16000,
            device: str = None):
        self.hidden_size = 512
        self.ckpt_path = ckpt_path
        self.config_path = config_path

        super().__init__(sample_rate=sample_rate, device=device)

    def load_model(self):
        from hificodec.vqvae import VQVAE
        model = VQVAE(
            self.config_path,
            self.ckpt_path,
            with_encoder=True
        )
        return model

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        acoustic_token = self.model.encode(wav)
        quant_emb = self.model.quantizer.embed(acoustic_token)
        latent = quant_emb.squeeze(0).transpose(0, 1)
        return latent

    def recon_wav(self, wav, n_q=None):
        acoustic_token = self.model.encode(wav)
        quant_emb = self.model.quantizer.embed(acoustic_token)
        recon_wav = self.model.generator(quant_emb).squeeze(0)
        return recon_wav


# 20250125 update
class StableCodecWrapper(AudioTokenizer):
    def __init__(self, ckpt_path: str = None, config_path: str = None, sample_rate: int = 16000, device: str = None):
        from huggingface_hub import hf_hub_download
        if ckpt_path == None:
            ckpt_path = hf_hub_download(
                repo_id="stabilityai/stable-codec-speech-16k",
                filename="model.safetensors",
                token = HF_TOKEN
            )
        if config_path == None:
            config_path = hf_hub_download(
                repo_id="stabilityai/stable-codec-speech-16k",
                filename="model_config.json",
                token = HF_TOKEN
            )
        
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.sample_rate = sample_rate
        self.device = device if device is not None else "cpu"
        self.codebook_size = 2 ** 17
        from stable_codec import StableCodec
        self.model = StableCodec(
            model_config_path=self.config_path,
            ckpt_path=self.ckpt_path,
            device=torch.device(self.device)
        )

    def load(self, audio_file: str,normalize = True):
        audio, sample_rate = torchaudio.load(audio_file)
        audio = self.model.model.preprocess_audio_for_encoder(audio.to(self.device), sample_rate)
        if normalize:
            audio = self.model.volume_norm(audio.squeeze(0)).unsqueeze(0)
        wav = audio.squeeze(0)
        return wav
    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        latents, tokens = self.model.encode(wav.unsqueeze(0))
        return latents
    
    def get_code(self, wav, n_q = None):
        latents, tokens = self.model.encode(wav.unsqueeze(0),posthoc_bottleneck = True)
        # code = tokens[:n_q]
        code = torch.cat(tokens, dim=0)
        code = code.permute(2, 0, 1)
        return code

    def recon_wav(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        # latents:[1, 6, 36] tokens:[1, 36, 1]
        latents, tokens = self.model.encode(wav.unsqueeze(0),posthoc_bottleneck = True)
        decoded_audio = self.model.decode(tokens,posthoc_bottleneck = True)
        recon = decoded_audio.squeeze(0)
        return recon

class StableCodecBaseWrapper(StableCodecWrapper):
    def __init__(self, ckpt_path: str = None, config_path: str = None, sample_rate: int = 16000, device: str = None):
        from huggingface_hub import hf_hub_download
        if ckpt_path == None:
            ckpt_path = hf_hub_download(
                repo_id="stabilityai/stable-codec-speech-16k-base",
                filename="model.safetensors",
                token = HF_TOKEN
            )
        if config_path == None:
            config_path = hf_hub_download(
                repo_id="stabilityai/stable-codec-speech-16k-base",
                filename="model_config.json",
                token = HF_TOKEN
            )
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.sample_rate = sample_rate
        self.device = device if device is not None else "cpu"
        self.codebook_size = 2 ** 17
        from stable_codec import StableCodec
        self.model = StableCodec(
            model_config_path=self.config_path,
            ckpt_path=self.ckpt_path,
            device=torch.device(self.device)
        )


class WavTokenizer600Wrapper(AudioTokenizer):
    def __init__(self,sample_rate=24000, bandwidth_id=0, device="cpu"):
        self.hidden_size = 512
        self.sample_rate = sample_rate
        self.device = device
        self.bandwidth_id = torch.tensor([bandwidth_id], device=self.device)
        self.codebook_size = 4096
        from huggingface_hub import hf_hub_download
        # self.ckpt_path = hf_hub_download(
        #     repo_id="novateur/WavTokenizer-large-unify-40token",
        #     filename="wavtokenizer_large_unify_600_24k.ckpt"
        # )


        self.ckpt_path = hf_hub_download(
            repo_id="novateur/WavTokenizer",
            filename="WavTokenizer_small_600_24k_4096.ckpt"
        )
        self.config_path = "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        super().__init__(sample_rate=sample_rate, device=device)

    def load_model(self):
        from WavTokenizer.decoder.pretrained import WavTokenizer
        model = WavTokenizer.from_pretrained0802(self.config_path, self.ckpt_path)
        model = model.to(self.device)
        return model

    def get_code(self, wav, n_q = None):
        features, codes = self.model.encode_infer(wav.to(self.device), bandwidth_id=self.bandwidth_id)
        code = codes
        return code

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        features, codes = self.model.encode_infer(wav.to(self.device), bandwidth_id=self.bandwidth_id)
        return features

    def recon_wav(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        features, codes = self.model.encode_infer(wav.to(self.device), bandwidth_id=self.bandwidth_id)
        audio_out = self.model.decode(features, bandwidth_id=self.bandwidth_id)
        return audio_out

class WavTokenizer320Wrapper(WavTokenizer600Wrapper):
    def __init__(self, sample_rate=24000, bandwidth_id=0, device="cpu"):
        self.hidden_size = 512

        self.sample_rate = sample_rate
        self.device = device
        self.bandwidth_id = torch.tensor([bandwidth_id], device=self.device)
        self.codebook_size = 4096

        from huggingface_hub import hf_hub_download
        # self.ckpt_path = hf_hub_download(
        #     repo_id="novateur/WavTokenizer-large-speech-75token",
        #     filename="wavtokenizer_large_speech_320_24k.ckpt"
        # )

        self.ckpt_path = hf_hub_download(
            repo_id="novateur/WavTokenizer",
            filename="WavTokenizer_small_320_24k_4096.ckpt"
        )
        self.config_path = "WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        AudioTokenizer.__init__(self,sample_rate=sample_rate, device=device)


class BigCodecWrapper(AudioTokenizer):
    def __init__(self, ckpt_path = None, sample_rate=16000, device="cpu"):
        self.hidden_size = 512
        if ckpt_path == None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id="Alethia/BigCodec",filename="bigcodec.pt")
            
        self.ckpt_path = ckpt_path
        self.sample_rate = sample_rate
        self.device = device
        self.encoder = None
        self.decoder = None
        self.codebook_size=8192

        from BigCodec.vq.codec_encoder import CodecEncoder
        from BigCodec.vq.codec_decoder import CodecDecoder

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.encoder = CodecEncoder()
        self.encoder.load_state_dict(ckpt["CodecEnc"])
        self.encoder.eval()
        self.encoder.to(self.device)

        self.decoder = CodecDecoder()
        self.decoder.load_state_dict(ckpt["generator"])
        self.decoder.eval()
        self.decoder.to(self.device)

    def get_latent(self, wav: torch.Tensor, n_q=8) -> torch.Tensor:
        vq_emb = self.encoder(wav.unsqueeze(1).to(self.device))
        vq_post_emb, vq_code, _ = self.decoder(vq_emb, vq=True)
        return vq_post_emb

    def get_code(self, wav, n_q = None):
        vq_emb = self.encoder(wav.unsqueeze(1).to(self.device))
        vq_post_emb, vq_code, _ = self.decoder(vq_emb, vq=True)
        code = vq_code
        return  code
        
    
    def load(self, audio_file: str):
        wav,sr = torchaudio.load(audio_file)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wav = wav.unsqueeze(0)
        wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            wav = resampler(wav)
        wav = wav.squeeze(0)
        return wav.to(self.device)

    def recon_wav(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        vq_emb = self.encoder(wav.unsqueeze(1).to(self.device))
        vq_post_emb, vq_code, _ = self.decoder(vq_emb, vq=True)
        recon = self.decoder(vq_post_emb, vq=False).squeeze(0).detach().cpu()
        return recon


class MimiWrapper(AudioTokenizer):
    def __init__(self, model_name_or_path="kyutai/mimi", device="cpu", sample_rate=24000):
        self.hidden_size = 512
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.codebook_size = 2048

        from transformers import MimiModel, AutoFeatureExtractor
        self.model = MimiModel.from_pretrained(self.model_name_or_path)
        
        # self.model.config.num_quantizers = 32
        self.model = self.model.eval().to(self.device)
        # EncodecFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
        self.sample_rate = self.feature_extractor.sampling_rate

    def load(self, audio_file: str):
        wav, sr = torchaudio.load(audio_file)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)
            wav = resampler(wav)
            sr = self.feature_extractor.sampling_rate
        audio_array = wav.squeeze(0).numpy()
        inputs = self.feature_extractor(raw_audio=audio_array, sampling_rate=sr, return_tensors="pt")
        wav = inputs["input_values"].squeeze(0)
        return wav.to(self.device)
    
    def get_code(self, wav, n_q = None):
        encoder_outputs = self.model.encode(wav.unsqueeze(0),num_quantizers = n_q)
        # [1,n_q,seq]
        codes = encoder_outputs.audio_codes
        code = codes
        return code        

    def get_latent(self, wav: torch.Tensor, sr: int, n_q=8) -> torch.Tensor:
        encoder_outputs = self.model.encode(wav)
        return encoder_outputs.audio_codes

    def recon_wav(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        encoder_outputs = self.model.encode(wav.unsqueeze(0),num_quantizers = n_q)
        codes = encoder_outputs.audio_codes
        audio_values = self.model.decode(codes)[0].cpu()
        recon_wav = audio_values.squeeze(0)
        return recon_wav



class XCodec2Wrapper(AudioTokenizer):
    def __init__(self, model_name_or_path="HKUST-Audio/xcodec2", sample_rate=16000, device="cpu"):
        self.hidden_size = 512
        self.model_path = model_name_or_path
        self.sample_rate = sample_rate
        self.device = device
        self.codebook_size = 65536
        super().__init__(sample_rate=sample_rate, device=device)

    def load_model(self):
        from xcodec2.modeling_xcodec2 import XCodec2Model
        model = XCodec2Model.from_pretrained(self.model_path)
        model = model.eval().to(self.device)
        return model

    def get_code(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        code = self.model.encode_code(input_waveform=wav.to(self.device))
        code = code
        return code

    def recon_wav(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        code = self.model.encode_code(input_waveform=wav.to(self.device))
        recon = self.model.decode_code(code).cpu().squeeze(0)
        return recon



class BigVGAN2Wrapper(AudioTokenizer):
    def __init__(self,device="cpu"):
        self.hidden_size = 512
        self.sample_rate = 22050
        self.device = device

        from BigVGAN import bigvgan
        model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x', use_cuda_kernel=False)
        model.remove_weight_norm()
        self.model = model.eval().to(device)
        # instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
        


    def recon_wav(self, wav: torch.Tensor, n_q=None) -> torch.Tensor:
        from BigVGAN.meldataset import get_mel_spectrogram

        mel = get_mel_spectrogram(wav, self.model.h).to(self.device) # mel is FloatTensor with shape [B(1), C_mel, T_frame]

        with torch.inference_mode():
            wav_gen = self.model(mel) 
        recon = wav_gen.squeeze(0)
        return recon

def build_model(model_name, device):
    if model_name == 'Hubert':
        return HubertWrapper(device=device)
    elif model_name == 'w2v_bert2':
        return HubertWrapper(device=device, pre_trained_folder="facebook/w2v-bert-2.0", hidden_size=1024)
    elif model_name == 'wavlm_plus':
        return HubertWrapper(device=device, pre_trained_folder="microsoft/wavlm-base-plus", hidden_size=768)
    elif model_name == 'SpeachTokenzier':
        return SpeachTokenzierWrapper(
            device=device,
        )
    elif model_name == 'DAC_16k':
        return DAC_16kWrapper(device=device)
    elif model_name == 'DAC_24k':
        return DAC_24kWrapper(device=device)
    elif model_name == 'DAC_44k':
        return DAC_44kWrapper(device=device)
    elif model_name == 'Encodec':
        return EncodecWrapper(device=device)

    elif model_name == 'Encodec_uniaudio':
        return Encodec_uniaudioWrapper(
            device=device,
            pre_trained_folder="/universal_model"
        )

    elif model_name == 'FACodec':
        return FACodecWrapper(
            device=device,
        )
    elif model_name == 'HIFICodec':
        return HIFICodecWrapper(
            ckpt_path="/HiFi-Codec-16k-320d",
            config_path='hificodec/config_16k_320d.json',
            device=device,
        )
    elif model_name == 'HIFICodec_universal':
        return HIFICodecWrapper(
            ckpt_path="/HiFi-Codec-16k-320d-large-universal",
            config_path="hificodec/config_16k_320d.json",
            device=device,
        )
    elif model_name == 'SemanticCodec_700bps':
        return SemantiCodecWrapper(
            bps = 700,
            device=device,
        )
    elif model_name == 'SemanticCodec_680bps':
        return SemantiCodecWrapper(
            bps = 680,
            device=device,
        )
    elif model_name == 'Xcodec':
        return XcodecWrapper(
            device=device
        )
    elif model_name == 'Xcodec2':
        return XCodec2Wrapper(
            device=device,
        )
    elif model_name == 'WavTokenizer600':
        return WavTokenizer600Wrapper(
            device=device,
        )
    elif model_name == 'WavTokenizer320':
        return WavTokenizer320Wrapper(
            device=device,
        )
    elif model_name == 'BigCodec':
        return BigCodecWrapper(
            device=device,
        )
    elif model_name == 'Mimi':
        return MimiWrapper(
            device=device,
        )
    elif model_name == 'StableCodec':
        return StableCodecWrapper(
            device=device,
        )
    elif model_name == 'StableCodec_base':
        return StableCodecBaseWrapper(
            device=device,
        )
    elif model_name == 'StableCodec_base_400bps':
        sc =  StableCodecBaseWrapper(
            device=device,
        )
        sc.codebook_size = 46656
        sc.model.set_posthoc_bottleneck("1x46656_400bps")
        return sc
    elif model_name == 'StableCodec_base_700bps':
        sc =  StableCodecBaseWrapper(
            device=device,
        )
        sc.codebook_size = 15625
        sc.model.set_posthoc_bottleneck("2x15625_700bps")
        return sc    
    elif model_name == 'StableCodec_base_1000bps':
        sc =  StableCodecBaseWrapper(
            device=device,
        )
        sc.codebook_size = 729
        sc.model.set_posthoc_bottleneck("4x729_1000bps")
        return sc
    elif model_name == "BigVGAN2":
        return BigVGAN2Wrapper(
            device=device,
        )       
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":

    Model_names = [
        # "DAC_16k",
        # "DAC_24k",
        # "DAC_44k",
        # "Encodec",

        # "SemanticCodec_680bps",
        # "SemanticCodec_700bps",
        # "Xcodec"

        # "SpeachTokenzier",
        # "Xcodec2",
        # "WavTokenizer600",
        # "WavTokenizer320",
        # "Mimi",
        # "BigCodec",

        # "StableCodec",
        # "StableCodec_base"
        # "StableCodec_base_400bps",
        # "StableCodec_base_700bps",
        # "StableCodec_base_1000bps"

        "BigVGAN2"
    ]

    audio_file = "LJ001-0001.wav"
    DEVICE = "cuda:0"  # or "cpu", depending on your setup
    N_Q = 2
    WAV_SEC = 10
    for model_name in Model_names:
        tokenizer = build_model(model_name, device=DEVICE)
        

        # wav = tokenizer.load(audio_file)

        wav = torch.randn([1,tokenizer.sample_rate * WAV_SEC]).to(DEVICE)


        recon = tokenizer.recon_wav(wav, n_q=8)
        print(f"recon.shape = {recon.shape}  wav.shape = {wav.shape}")
        print(f"recon = {recon}")

        print(f"Loaded audio for {model_name}. Shape: {wav.shape}")
        code = tokenizer.get_code(wav, n_q=N_Q)
        
        print(f"=======  Testing model: {model_name} =======")

        print(f"Code shape: {code.shape}, Code max: {torch.max(code)} tokenizer.codebook_size = {tokenizer.codebook_size}")

        import numpy as np
        bitrate = code.shape[-1] * np.ceil(np.log2(tokenizer.codebook_size)) * N_Q / WAV_SEC
        # print(f"code = {code}")
        print(f"Bitrate = {bitrate} codebook_size = {tokenizer.codebook_size}  token rate = {code.shape[-1] / WAV_SEC} ")

        print(f"===========================")




    import pdb;pdb.set_trace()
    # code = tokenizer.get_code(wav, n_q=4)
    # print(f"code.shape = {code.shape} code_max = {torch.max(code)}")


    # embeddings = tokenizer.get_embedding(wav, n_q=8)
    # print(embeddings)
    #
    # embeddings = tokenizer.get_embedding(wav, n_q=4)
    # print(embeddings)

    # latent = tokenizer.get_latent(wav, n_q=8)
    # embeddings = tokenizer.get_embedding(wav, n_q=8)
    # print(embeddings.shape)
    # print(latent.shape)