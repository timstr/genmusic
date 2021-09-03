import torchaudio
import torch
import glob
import sys
import os
import numpy as np


def main(source_path, target_path):
    filenames = list(sorted(glob.glob(f"{source_path}/*")))
    if len(filenames) == 0:
        print(f'No files were found in "{source_path}"')
        return
    for fn in filenames:
        _, tail = os.path.split(fn)
        tail_name, tail_ext = tail.split(".")
        new_path = os.path.join(target_path, f"{tail_name}.flac")

        try:
            audio, _ = torchaudio.load(fn)
        except:
            continue
        channels, length = audio.shape
        if channels != 2:
            raise Exception(f'"{fn}" needs to be stereo audio.')
        print(f'resampling "{fn}"')

        max_amp = torch.max(torch.abs(audio))
        max_permitted_amp = 0.99
        if max_amp >= max_permitted_amp:
            print(f"Warning: amplitude is too high and is being scaled")
            audio *= max_permitted_amp / max_amp

        if length == 65536:
            print(f'copying "{fn}"')
            torchaudio.save(filepath=new_path, src=audio, sample_rate=32000)
            continue

        audio_resampled = torch.zeros((2, 65536), dtype=torch.float)
        for i in range(2):
            channel = audio[i]
            channel_resampled = np.interp(
                x=np.linspace(start=0.0, stop=length, num=65536, endpoint=False),
                xp=np.linspace(start=0.0, stop=length, num=length, endpoint=False),
                fp=channel.numpy(),
            )
            channel_resampled = torch.tensor(channel_resampled)
            assert channel_resampled.shape == (65536,)
            audio_resampled[i] = channel_resampled

        assert audio_resampled.shape == (2, 65536)

        torchaudio.save(filepath=new_path, src=audio_resampled, sample_rate=32000)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} path/to/sources path/to/destination")
        exit(1)
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    if not os.path.exists(source_path):
        raise Exception(f'"{source_path}" is not a valid path')
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    main(source_path, target_path)
