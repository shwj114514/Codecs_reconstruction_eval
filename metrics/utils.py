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




import logging
import os
import functools
import math
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import glob
import matplotlib.pylab as plt


def get_paths_with_cache(search_path, cache_path=None):
    out_paths = None
    if cache_path != None and os.path.exists(cache_path):
        out_paths = torch.load(cache_path)
    else:
        path = Path(search_path)
        out_paths = find_audio_files(path, ['.wav', '.m4a', '.mp3'])
        if cache_path is not None:
            print("Building cache..")
            torch.save(out_paths, cache_path)
    return out_paths


def find_audio_files(folder_path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(folder_path, '**', f'*{suffix}'), recursive=True))
    return files


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={},batch_audios ={},batch_f0={}, audio_sampling_rate=44100):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)

    for k, v in batch_audios.items():
        for nb in range(v.shape[0]):
            writer.add_audio(
                    f"{k}/sample_{nb}.wav", v[nb].cpu(), global_step,audio_sampling_rate
                )
            
    # for k, v in batch_f0.items():
    #     for nb in range(v.shape[0]):
    #         writer.add_audio(
    #                 f"{k}/sample_{nb}.wav", v[nb].cpu(), global_step,audio_sampling_rate
    #             )



MATPLOTLIB_FLAG = False

def plot_f0(predict_f0, gd_f0, f0_min=0.0, f0_max=600.0):
    # predict_f0 naaray(len)     f0_min=50.0
    num_samples = len(predict_f0)
    sampling_rate = 44100  # 每秒采样点数
    time_axis = np.linspace(0, num_samples*512 / sampling_rate, num_samples, endpoint=False)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time_axis, predict_f0, label='Predicted f0', color='blue', linestyle='-', linewidth=2)
    ax.plot(time_axis, gd_f0, label='Ground Truth f0', color='red', linestyle='--', linewidth=2)

    ax.set_ylim(f0_min, f0_max)

    ax.set_title('Fundamental Frequency (F0) Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend()
    # plt的数据放入到numpy
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


logger = logging


def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    name_key = (lambda _f: int(re.compile('model-(\d+)\.pt').match(_f).group(1)))
    time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')],
                                 key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
        (x_sorted('model')[:-n_ckpts_to_keep])]
    del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]


# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(
                bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.factor = factor
        if use_conv:
            ksize = 5
            pad = 2
            self.conv = nn.Conv1d(self.channels, self.out_channels, ksize, padding=pad)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4, ksize=5, pad=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        stride = factor
        if use_conv:
            self.op = nn.Conv1d(
                self.channels, self.out_channels, ksize, stride=stride, padding=pad
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            up=False,
            down=False,
            kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


DEFAULT_MEL_NORM_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/mel_norms.pth')


class TorchMelSpectrogram(nn.Module):
    def __init__(
            self,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            mel_fmin=0,
            mel_fmax=8000,
            sampling_rate=22050,
            normalize=False,
            mel_norm_file=DEFAULT_MEL_NORM_FILE
    ):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        if torch.backends.mps.is_available():
            inp = inp.to('cpu')
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel


# https://github.com/descriptinc/audiotools/blob/7776c296c711db90176a63ff808c26e0ee087263/audiotools/core/audio_signal.py
def get_window(window_type: str, window_length: int, device: str):
    """Wrapper around scipy.signal.get_window so one can also get the
    popular sqrt-hann window. This function caches for efficiency
    using functools.lru\_cache.

    Parameters
    ----------
    window_type : str
        Type of window to get
    window_length : int
        Length of the window
    device : str
        Device to put window onto.

    Returns
    -------
    torch.Tensor
        Window returned by scipy.signal.get_window, as a tensor.
    """
    if window_length ==None:
        window_length =2048
    from scipy import signal

    if window_type == "average":
        window = np.ones(window_length) / window_length
    elif window_type == "sqrt_hann":
        window = np.sqrt(signal.get_window("hann", window_length))
    else:
        window = signal.get_window(window_type, window_length)
    window = torch.from_numpy(window).to(device).float()
    return window


def compute_stft_padding(
        signal_length: int, window_length: int, hop_length: int, match_stride: bool
):
    """Compute how the STFT should be padded, based on match\_stride.

    Parameters
    ----------
    window_length : int
        Window length of STFT.
    hop_length : int
        Hop length of STFT.
    match_stride : bool
        Whether or not to match stride, making the STFT have the same alignment as
        convolutional layers.

    Returns
    -------
    tuple
        Amount to pad on either side of audio.
    """

    if match_stride:
        assert (
                hop_length == window_length // 4
        ), "For match_stride, hop must equal n_fft // 4"
        right_pad = math.ceil(signal_length / hop_length) * hop_length - signal_length
        pad = (window_length - hop_length) // 2
    else:
        right_pad = 0
        pad = 0

    return right_pad, pad


def stft(
        wav: torch.Tensor,  # (batch_size, num_channels, num_samples)
        window_length: int = 2048,
        hop_length: int = 512,
        window_type: str = 'hann',
        match_stride: bool = False,
        padding_type: str = 'reflect',
):
    """Computes the short-time Fourier transform of the audio data,
    with specified STFT parameters.

    Parameters
    ----------
    window_length : int, optional
        Window length of STFT, by default ``0.032 * self.sample_rate``.
    hop_length : int, optional
        Hop length of STFT, by default ``window_length // 4``.
    window_type : str, optional
        Type of window to use, by default ``sqrt\_hann``.
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False
    padding_type : str, optional
        Type of padding to use, by default 'reflect'

    Returns
    -------
    torch.Tensor
        STFT of audio data.

    Compute the STFT of an AudioSignal:

    """
    window = get_window(window_type, window_length, wav.device)
    window = window.to(wav.device)

    right_pad, pad = compute_stft_padding(
        signal_length=wav.shape[-1],
        window_length=window_length,
        hop_length=hop_length,
        match_stride=match_stride
    )
    wav = torch.nn.functional.pad(
        wav, (pad, pad + right_pad), padding_type
    )
    # import pdb;pdb.set_trace()
    stft_data = torch.stft(
        wav.reshape(-1, wav.shape[-1]),
        n_fft=window_length,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
    )
    _, nf, nt = stft_data.shape
    # stft_data = stft_data.reshape(self.batch_size, self.num_channels, nf, nt)
    stft_data = stft_data.reshape(wav.shape[0], wav.shape[1], nf, nt)

    if match_stride:
        # Drop first two and last two frames, which are added
        # because of padding. Now num_frames * hop_length = num_samples.
        stft_data = stft_data[..., 2:-2]
    return stft_data



def get_mel_filters(
    sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None
):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Parameters
    ----------
    sr : int
        Sample rate of audio
    n_fft : int
        Number of FFT bins
    n_mels : int
        Number of mels
    fmin : float, optional
        Lowest frequency, in Hz, by default 0.0
    fmax : float, optional
        Highest frequency, by default None

    Returns
    -------
    np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    """
    from librosa.filters import mel as librosa_mel_fn

    return librosa_mel_fn(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )



def mel_spectrogram(
        wav:torch.Tensor,
        n_mels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: float = None,
        **kwargs
):
    """Computes a Mel spectrogram.

    Parameters
    ----------
    n_mels : int, optional
        Number of mels, by default 80
    mel_fmin : float, optional
        Lowest frequency, in Hz, by default 0.0
    mel_fmax : float, optional
        Highest frequency, by default None
    kwargs : dict, optional
        Keyword arguments to self.stft().

    Returns
    -------
    torch.Tensor [shape=(batch, channels, mels, time)]
        Mel spectrogram.
    """
    stft_data = stft(wav= wav)
    magnitude = torch.abs(stft_data)

    nf = magnitude.shape[2]
    mel_basis = get_mel_filters(
        sr=44100,
        n_fft=2 * (nf - 1),
        n_mels=n_mels,
        fmin=mel_fmin,
        fmax=mel_fmax,
    )
    mel_basis = torch.from_numpy(mel_basis).to(wav.device)

    mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
    mel_spectrogram = mel_spectrogram.transpose(-1, 2)
    return mel_spectrogram


# ttts mel_extractor
def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))

class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=44100, n_fft=1024, hop_length=512, n_mels=80, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features



if __name__ == "__main__":
    wav = torch.randn(1,1,44100)
    mel_spectrogram = mel_spectrogram(wav)
    plt_data = plot_spectrogram_to_numpy(mel_spectrogram.squeeze().detach().cpu())
    import pdb;pdb.set_trace()