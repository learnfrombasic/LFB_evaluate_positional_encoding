from typing import Union

import numpy as np
import torch
import av
import io
import librosa
import torch.nn.functional as F
import torchaudio.transforms as T


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalize audio waveform to be between -1 and 1.
    """
    waveform = waveform - torch.mean(waveform)
    return waveform / torch.max(torch.abs(waveform)) + 1.0e-6


def shift_waveform(waveform: np.ndarray, shift: int) -> np.ndarray:
    """
    Shifts the waveform by a specified number of samples.

    Parameters:
    waveform (numpy array): The input audio signal.
    shift (int): The number of samples to shift. Positive values shift right, negative values shift left.

    Returns:
    numpy array: The shifted waveform.
    """
    return np.roll(waveform, shift)


def pad_or_truncate(
    x: Union[np.ndarray, torch.Tensor], audio_length: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Pad or truncate the audio waveform to a fixed length.

    Args:
        x (torch.Tensor): The audio waveform tensor.
        audio_length (int): The desired length of the audio.

    Returns:
        torch.Tensor: The padded or truncated waveform.

    Examples:
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> audio_length = 7
        >>> pad_or_truncate(x, audio_length)
        tensor([1, 2, 3, 4, 5, 0, 0])

        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> audio_length = 3
        >>> pad_or_truncate(x, audio_length)
        tensor([1, 2, 3])

        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> audio_length = 5
        >>> pad_or_truncate(x, audio_length)
        tensor([1, 2, 3, 4, 5])
    """
    if isinstance(x, torch.Tensor):
        padding = audio_length - x.size(0)
        if padding > 0:
            return torch.nn.functional.pad(x, (0, padding))
        else:
            return x[:audio_length]
    else:
        if len(x) <= audio_length:
            return np.concatenate(
                (x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0
            )
        else:
            return x[0:audio_length]


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array of int16 type to float32 type.

    This function takes an input array of int16 type and performs a conversion to float32 type.
    The conversion is done by dividing each element of the input array by 32767.0 and then
    casting the result to float32 type.

    Args:self
        x (np.ndarray): Input array of int16 type.

    Returns:
        np.ndarray: Converted array of float32 type.
    """
    return (x / 32767.0).astype(np.float32)


def add_white_noise(
    waveform: Union[np.ndarray, torch.Tensor], noise_level: float = 0.01
) -> Union[np.ndarray, torch.Tensor]:
    """
    Adds white noise to the audio waveform.

    Args:
        waveform: The input audio signal.
        noise_level: The scale of the noise.

    Returns:
        The noisy waveform.
    """
    if isinstance(waveform, torch.Tensor):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    else:
        noise = np.random.randn(*waveform.shape) * noise_level
        return waveform + noise


def add_brown_noise(
    waveform: Union[np.ndarray, torch.Tensor], noise_level: float = 0.01
) -> Union[np.ndarray, torch.Tensor]:
    """
    Adds brown noise to the audio waveform (integration of white noise).

    Args:
        waveform: The input audio signal.
        noise_level: The scale of the noise.

    Returns:
        The noisy waveform.
    """
    if isinstance(waveform, torch.Tensor):
        noise = torch.randn_like(waveform)
        brown_noise = torch.cumsum(noise, dim=-1)
        # Normalize to prevent exploding values
        brown_noise = brown_noise / torch.max(torch.abs(brown_noise)) * noise_level
        return waveform + brown_noise
    else:
        noise = np.random.randn(*waveform.shape)
        brown_noise = np.cumsum(noise, axis=-1)
        brown_noise = brown_noise / np.max(np.abs(brown_noise)) * noise_level
        return waveform + brown_noise


def time_stretch(
    waveform: Union[np.ndarray, torch.Tensor], rate: float
) -> Union[np.ndarray, torch.Tensor]:
    """
    Time-stretches an audio waveform without changing its pitch.

    Args:
        waveform: The input audio signal.
        rate: Stretch factor. If rate > 1, the audio is sped up. If rate < 1, the audio is slowed down.

    Returns:
        The time-stretched waveform.
    """
    is_tensor = isinstance(waveform, torch.Tensor)
    if is_tensor:
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    stretched = librosa.effects.time_stretch(y=wav_np, rate=rate)

    if is_tensor:
        return torch.from_numpy(stretched)
    return stretched


def pitch_shift(
    waveform: Union[np.ndarray, torch.Tensor], sr: int, n_steps: float
) -> Union[np.ndarray, torch.Tensor]:
    """
    Pitch-shifts an audio waveform without changing its duration.

    Args:
        waveform: The input audio signal.
        sr: The sample rate of the audio.
        n_steps: The number of fractional half-steps to shift. Positive values shift pitch up, negative shift down.

    Returns:
        The pitch-shifted waveform.
    """
    is_tensor = isinstance(waveform, torch.Tensor)
    if is_tensor:
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    shifted = librosa.effects.pitch_shift(y=wav_np, sr=sr, n_steps=n_steps)

    if is_tensor:
        return torch.from_numpy(shifted)
    return shifted
