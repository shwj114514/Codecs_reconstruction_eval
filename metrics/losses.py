import typing
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from utils import mel_spectrogram,stft


def envelope_loss(y, y_hat):
    """
        copied RefineGAN `generator_envelope_loss` from
        https://github.com/fishaudio/fish-diffusion/blob/018413486e4fdf6d5b0f74266494c32ec0b816c5/fish_diffusion/archs/hifisinger/hifisinger_v2.py#L100
    """

    def extract_envelope(signal, kernel_size=100, stride=50):
        envelope = F.max_pool1d(signal, kernel_size=kernel_size, stride=stride)
        return envelope

    y_envelope = extract_envelope(y)
    y_hat_envelope = extract_envelope(y_hat)

    y_reverse_envelope = extract_envelope(-y)
    y_hat_reverse_envelope = extract_envelope(-y_hat)

    loss_envelope = F.l1_loss(y_envelope, y_hat_envelope) + F.l1_loss(
        y_reverse_envelope, y_hat_reverse_envelope
    )

    return loss_envelope


class f0Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_lf0, LF0):
        loss_f0 = F.mse_loss(predict_lf0, LF0)
        return loss_f0


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
            self,
            scaling: int = True,
            reduction: str = "mean",
            zero_mean: int = True,
            clip_min: int = None,
            weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, references, estimates):
        eps = 1e-8
        # nb, nc, nt
        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references ** 2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


@dataclass
class StftParams:
    window_length: int
    hop_length: int
    match_stride: bool
    window_type: str


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
            self,
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            window_type: str = 'hann', # None
    ):
        super().__init__()
        self.stft_params = [
            StftParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x, y):
        """Computes multi-scale STFT between an estimate and a reference signal.

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0

 
        for s in self.stft_params:
            x_stft_data = stft(x,s.window_length, s.hop_length, s.window_type)
            y_stft_data = stft(y,s.window_length, s.hop_length, s.window_type)
            x_magnitude = torch.abs(x_stft_data)
            y_magnitude = torch.abs(y_stft_data)
            loss += self.log_weight * self.loss_fn(
                x_magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_magnitude, y_magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
            self,
            n_mels: List[int] = [150, 80],
            window_lengths: List[int] = [2048, 512],
            loss_fn: typing.Callable = nn.L1Loss(),
            clamp_eps: float = 1e-5,
            mag_weight: float = 1.0,
            log_weight: float = 1.0,
            pow: float = 2.0,
            weight: float = 1.0,
            match_stride: bool = False,
            mel_fmin: List[float] = [0.0, 0.0],
            mel_fmax: List[float] = [None, None],
            window_type: str = 'hann',
    ):
        super().__init__()
        self.stft_params = [
            StftParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x, y):
        """Computes mel loss between an estimate (x) and a reference (y)
        x,y [B,1,16758]
        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
                self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = mel_spectrogram(wav=x, n_mels=n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = mel_spectrogram(wav=y, n_mels=n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


if __name__ == '__main__':
    mel_loss = MelSpectrogramLoss()
    x = torch.randn(1, 44100)
    y = torch.randn(1, 44100)
    print(mel_loss(x, y))