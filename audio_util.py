import glob
import os
import torch
import torchaudio
import random

from progress_bar import progress_bar


def load_audio_clips(glob_path, fade_length_samples=256):
    songs = []
    filenames = sorted(glob.glob(glob_path))
    # filenames = sorted(glob.glob("sound/adventure of a lifetime*.flac"))
    fade_weights = torch.linspace(0.0, 1.0, steps=fade_length_samples).unsqueeze(0)
    for i, filename in enumerate(filenames):
        progress_bar(i, len(filenames), "audio clips loaded")
        song_path, song_file = os.path.split(filename)
        song_name, song_ext = song_file.split(".")
        song_audio, sample_rate = torchaudio.load(filename)
        if song_audio.shape != (2, 65536):
            raise Exception(
                f'The file "{filename}" needs to be 2-channel stereo and 65536 sample long.'
            )
        song_audio[:, -fade_length_samples:] *= 1.0 - fade_weights
        song_audio[:, -fade_length_samples:] += fade_weights * song_audio[:, 0:1]
        # song_audio[:, :fade_length_samples] *= fade_weights
        # song_audio[:, -fade_length_samples:] *= 1.0 - fade_weights
        songs.append((song_audio, song_name))
    return songs


def random_audio_batch(batch_size, patch_size, audio_clips):
    assert isinstance(batch_size, int)
    assert isinstance(patch_size, int)
    assert isinstance(audio_clips, list)
    assert len(audio_clips) > 0
    clips_subset = []
    for _ in range(batch_size):
        clip, random_name = random.choice(audio_clips)
        assert isinstance(clip, torch.Tensor)
        assert len(clip.shape) == 2
        assert clip.shape[0] == 2
        assert isinstance(random_name, str)

        # clip_length = clip.shape[1]
        # assert patch_size <= clip_length
        # clip_twice = torch.cat([clip, clip], dim=1)
        # assert clip_twice.shape == (2, clip_length * 2)
        # random_offset = random.randrange(0, clip_length)
        # clip = clip_twice[:, random_offset : random_offset + patch_size]
        # assert clip.shape == (2, patch_size)

        clips_subset.append(clip)

    clips_batch = torch.stack(clips_subset, dim=0).to("cuda")

    assert clips_batch.shape == (batch_size, 2, patch_size,)

    # N = clips_batch.shape[2]

    # HACK
    # max_offset = N - patch_size
    # if max_offset == 0:
    #     random_crop_offset = 0
    # else:
    #     random_crop_offset = random.randrange(0, max_offset)

    # clips_batch = clips_batch[
    #     :, :, random_crop_offset : random_crop_offset + patch_size
    # ]

    return clips_batch


def sane_audio_loss(audio):
    assert isinstance(audio, torch.Tensor)
    mean = torch.mean(audio, dim=1, keepdim=True)
    zero_mean_audio = audio - mean
    mean_audio_amp = torch.mean(torch.abs(zero_mean_audio))
    return torch.mean(torch.abs(mean)) + torch.clamp(mean_audio_amp - 0.5, min=0.0)
