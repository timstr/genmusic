import glob
import os
import torch
import torchaudio
import random

from progress_bar import progress_bar


def load_audio_clips(glob_path):
    songs = []
    filenames = sorted(glob.glob(glob_path))
    # filenames = sorted(glob.glob("sound/adventure of a lifetime*.flac"))
    for i, filename in enumerate(filenames):
        progress_bar(i, len(filenames), "audio clips loaded")
        song_path, song_file = os.path.split(filename)
        song_name, song_ext = song_file.split(".")
        song_audio, sample_rate = torchaudio.load(filename)
        if song_audio.shape != (2, 65536):
            raise Exception(
                f'The file "{filename}" needs to be 2-channel stereo and 65536 sample long.'
            )
        songs.append((song_audio, song_name))
    return songs


def random_audio_batch(batch_size, patch_size, audio_clips):
    assert isinstance(batch_size, int)
    assert isinstance(patch_size, int)
    assert isinstance(audio_clips, list)
    clips_subset = []
    for _ in range(batch_size):
        random_clip, random_name = random.choice(audio_clips)
        assert isinstance(random_clip, torch.Tensor)
        assert len(random_clip.shape) == 2
        assert random_clip.shape[0] == 2
        assert isinstance(random_name, str)
        clips_subset.append(random_clip)

    clips_batch = torch.stack(clips_subset, dim=0).to("cuda")

    assert len(clips_batch.shape) == 3
    assert clips_batch.shape[:2] == (batch_size, 2)

    N = clips_batch.shape[2]

    max_offset = N - patch_size
    if max_offset == 0:
        random_crop_offset = 0
    else:
        random_crop_offset = random.randrange(0, max_offset)

    clips_batch = clips_batch[
        :, :, random_crop_offset : random_crop_offset + patch_size
    ]

    return clips_batch



def sane_audio_loss(audio):
    assert isinstance(audio, torch.Tensor)
    mean = torch.mean(audio, dim=1, keepdim=True)
    zero_mean_audio = audio - mean
    mean_audio_amp = torch.mean(torch.abs(zero_mean_audio))
    return torch.mean(torch.abs(mean)) + torch.clamp(mean_audio_amp - 0.5, min=0.0)