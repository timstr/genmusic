import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import random

from signal_processing import make_spectrogram
from audio_util import load_audio_clips, random_audio_batch, sane_audio_loss
from torch_utils import (
    Resample1d,
    Log,
    ResidualAdd1d,
    enable_log_layers,
    save_module,
)
from util import assert_eq, plt_screenshot
from progress_bar import progress_bar


class LossPlotter:
    def __init__(self, aggregation_interval, colour, num_quantiles):
        assert isinstance(aggregation_interval, int)
        r, g, b = colour
        assert isinstance(r, float)
        assert isinstance(g, float)
        assert isinstance(b, float)
        assert isinstance(num_quantiles, int)
        assert num_quantiles >= 1 and num_quantiles < aggregation_interval
        self._colour_r = r
        self._colour_g = g
        self._colour_b = b
        self._aggregation_interval = aggregation_interval
        self._colour = colour
        self._num_quantiles = num_quantiles
        self._values = [[] for _ in range(num_quantiles + 1)]
        self._acc = []

    def append(self, item):
        assert isinstance(item, float)
        self._acc.append(item)
        if len(self._acc) == self._aggregation_interval:
            q = np.linspace(0.0, 1.0, num=(self._num_quantiles + 1))
            qv = np.quantile(self._acc, q)
            for i in range(self._num_quantiles + 1):
                self._values[i].append(qv[i])
            self._acc = []

    def plot_to(self, plt_axis):
        colour_dark = (self._colour_r, self._colour_g, self._colour_b)
        if self._aggregation_interval > 1:
            x_min = self._aggregation_interval // 2
            x_stride = self._aggregation_interval
            x_count = len(self._values[0])
            x_values = range(x_min, x_min + x_count * x_stride, x_stride)
            for i in range(self._num_quantiles):
                t = i / self._num_quantiles
                t = 2.0 * min(t, 1.0 - t)
                c = (self._colour_r, self._colour_g, self._colour_b, t)
                plt_axis.fill_between(
                    x=x_values, y1=self._values[i], y2=self._values[i + 1], color=c
                )
        else:
            plt_axis.scatter(
                range(self._values[0]), self._items, s=1.0, color=colour_dark
            )


class Generator(nn.Module):
    def __init__(self, num_latent_features):
        super(Generator, self).__init__()
        assert isinstance(num_latent_features, int)

        self.num_latent_features = num_latent_features

        self.fully_connected = nn.Sequential(
            Log("generator fully connected 0"),
            nn.Linear(in_features=num_latent_features, out_features=256),
            # nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            Log("generator fully connected 1"),
            nn.Linear(in_features=256, out_features=32 * 32 * 64),
            Log("generator fully connected 2"),
        )

        self.spectral_convs = nn.Sequential(
            Log("generator spectral conv 0"),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=(6, 6),
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            Log("generator spectral conv 1"),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=(7, 7),
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            Log("generator spectral conv 2"),
        )

        self.window_size = 256

        self.window = nn.parameter.Parameter(
            data=torch.hann_window(self.window_size, periodic=True), requires_grad=False
        )

        self.temporal_convs = nn.Sequential(
            Log("generator temporal conv 0"),
            Resample1d(new_length=32768),
            ResidualAdd1d(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=31,
                        stride=1,
                        padding=15,
                        padding_mode="circular",
                    ),
                    nn.BatchNorm1d(num_features=16),
                    nn.LeakyReLU(),
                    nn.Conv1d(
                        in_channels=16,
                        out_channels=8,
                        kernel_size=31,
                        stride=1,
                        padding=15,
                        padding_mode="circular",
                    ),
                )
            ),
            Log("generator temporal conv 1"),
            Resample1d(new_length=65536),
            ResidualAdd1d(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=31,
                        stride=1,
                        padding=15,
                        padding_mode="circular",
                    ),
                    nn.BatchNorm1d(num_features=8),
                    nn.LeakyReLU(),
                    nn.Conv1d(
                        in_channels=8,
                        out_channels=2,
                        kernel_size=31,
                        stride=1,
                        padding=15,
                        padding_mode="circular",
                    ),
                )
            ),
            Log("generator temporal conv 2"),
        )

    def forward(self, latent_codes):
        B, D = latent_codes.shape
        assert_eq(D, self.num_latent_features)

        x0 = latent_codes
        x1 = self.fully_connected(x0)
        assert_eq(x1.shape, (B, 32 * 32 * 64))

        x2 = x1.reshape(B, 32, 32, 64)

        x3 = self.spectral_convs(x2)
        assert_eq(x3.shape, (B, 32, 129, 257))

        x4 = torch.complex(real=x3[:, :16], imag=x3[:, 16:])

        assert_eq(x4.shape, (B, 16, 129, 257))
        x5 = x4.reshape(B * 16, 129, 257)

        x6 = torch.istft(
            input=x5,
            n_fft=self.window_size,
            window=self.window,
            return_complex=False,
            onesided=True,
        )
        assert_eq(x6.shape, (B * 16, 16384))

        x7 = x6.reshape(B, 16, 16384)
        x8 = self.temporal_convs(x7)
        assert_eq(x8.shape, (B, 2, 65536))

        return x8


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.temporal_convs = nn.Sequential(
            Log("discriminator temporal conv 0"),
            nn.Conv1d(
                in_channels=2, out_channels=8, kernel_size=31, stride=2, padding=15
            ),
            nn.BatchNorm1d(num_features=8),
            nn.LeakyReLU(),
            Log("discriminator temporal conv 1"),
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=31, stride=2, padding=15
            ),
            Log("discriminator temporal conv 2"),
        )

        self.window_size = 256

        self.window = nn.parameter.Parameter(
            data=torch.hann_window(self.window_size, periodic=True), requires_grad=False
        )

        self.spectral_convs = nn.Sequential(
            Log("discriminator spectral conv 0"),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=(6, 6),
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            Log("discriminator spectral conv 1"),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=(7, 7),
            ),
            Log("discriminator spectral conv 2"),
        )

        self.fully_connected = nn.Sequential(
            Log("discriminator fully connected 1"),
            nn.Linear(in_features=(32 * 32 * 64), out_features=256),
            # nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(),
            Log("discriminator fully connected 2"),
            nn.Linear(in_features=256, out_features=1),
            Log("discriminator fully connected 3"),
        )

    def forward(self, audio_clips):
        B, C, N = audio_clips.shape
        assert_eq(C, 2)
        assert_eq(N, 65536)

        x0 = audio_clips
        x1 = self.temporal_convs(x0)
        assert_eq(x1.shape, (B, 16, 16384))
        x2 = x1.reshape(B * 16, 16384)
        x3 = torch.stft(
            input=x2,
            n_fft=self.window_size,
            window=self.window,
            return_complex=True,
            onesided=True,
        )
        assert_eq(x3.shape, (B * 16, 129, 257))
        x4 = x3.reshape(B, 16, 129, 257)
        x5 = torch.cat([torch.real(x4), torch.imag(x4)], dim=1)
        assert_eq(x5.shape, (B, 32, 129, 257))
        x6 = self.spectral_convs(x5)
        assert_eq(x6.shape, (B, 32, 32, 64))
        x7 = x6.reshape(B, 32 * 32 * 64)
        x8 = self.fully_connected(x7)
        assert_eq(x8.shape, (B, 1))

        return x8


# HACK to test networks
# enable_log_layers()
# d = Discriminator().cuda()
# s = d(torch.rand((1, 2, 65536), device="cuda"))
# g = Generator(num_latent_features=64).cuda()
# a = g(torch.rand((1, 64), device="cuda"))
# exit(-1)


def random_initial_vector(batch_size, num_features):
    return -1.0 + 2.0 * torch.randn(
        (batch_size, num_features), dtype=torch.float32, device="cuda"
    )
    # return -1.0 + 2.0 * torch.rand(
    #     (batch_size, num_features, num_samples), dtype=torch.float32, device="cuda"
    # )


def train(
    all_audio_clips,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    latent_features,
    patch_size,
):
    assert isinstance(all_audio_clips, list)
    assert isinstance(generator, nn.Module)
    assert isinstance(discriminator, nn.Module)
    assert isinstance(generator_optimizer, torch.optim.Optimizer)
    assert isinstance(discriminator_optimizer, torch.optim.Optimizer)
    assert isinstance(latent_features, int)
    assert isinstance(patch_size, int)

    parallel_batch_size = 8
    sequential_batch_size = 1

    n_critic = 5
    n_generator = 1
    n_all = n_critic + n_generator

    generator_loss_fool_acc = 0.0
    generator_loss_sane_acc = 0.0
    discriminator_loss_real_acc = 0.0
    discriminator_loss_fake_acc = 0.0

    # sys.stdout.write("  ")
    # sys.stdout.flush()

    for i_batch in range(sequential_batch_size * n_all):
        mode = (i_batch // sequential_batch_size) % n_all
        first_step = (i_batch % sequential_batch_size) == 0
        last_step = ((i_batch + 1) % sequential_batch_size) == 0

        training_discriminator = mode < n_critic
        training_generator = not training_discriminator

        with torch.no_grad():
            real_clips_batch = random_audio_batch(
                batch_size=parallel_batch_size,
                patch_size=patch_size,
                audio_clips=all_audio_clips,
            )

        initial_vector = random_initial_vector(
            batch_size=parallel_batch_size,
            num_features=latent_features,
        )

        fake_clips_batch = generator(initial_vector)

        # Assign scores to real and fake audio
        discriminator_real_predictions = discriminator(real_clips_batch)
        discriminator_fake_predictions = discriminator(fake_clips_batch)

        total_loss = 0.0

        if training_generator:
            # generator wants to maximize the fake score
            l_fool = -torch.mean(discriminator_fake_predictions)
            l_sane = sane_audio_loss(fake_clips_batch)
            total_loss += l_fool
            total_loss += l_sane
            model = generator
            optimizer = generator_optimizer
            generator_loss_fool_acc -= l_fool.detach().cpu().item()
            generator_loss_sane_acc += l_sane.detach().cpu().item()
        if training_discriminator:
            # discriminator wants to maximize the real score and minimize the fake score
            l_real = -torch.mean(discriminator_real_predictions)
            l_fake = torch.mean(discriminator_fake_predictions)
            l = l_real + l_fake
            total_loss += l
            model = discriminator
            optimizer = discriminator_optimizer
            discriminator_loss_real_acc -= l_real.detach().cpu().item()
            discriminator_loss_fake_acc += l_fake.detach().cpu().item()

        if first_step:
            optimizer.zero_grad()

        total_loss.backward()

        if last_step:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad[...] /= sequential_batch_size
            optimizer.step()

        if training_discriminator and last_step:
            parameter_limit = 0.05
            with torch.no_grad():
                for p in discriminator.parameters():
                    p.clamp_(min=-parameter_limit, max=parameter_limit)

        # console_animation_characters = "-\\|/"
        # sys.stdout.write("\b")
        # if i_batch + 1 < (sequential_batch_size * n_all):
        #     sys.stdout.write(
        #         console_animation_characters[
        #             i_batch % len(console_animation_characters)
        #         ]
        #     )
        # sys.stdout.flush()

    generator_loss_fool_avg = generator_loss_fool_acc / (
        sequential_batch_size * n_generator
    )
    generator_loss_sane_avg = generator_loss_sane_acc / (
        sequential_batch_size * n_generator
    )
    discriminator_loss_real_avg = discriminator_loss_real_acc / (
        sequential_batch_size * n_critic
    )
    discriminator_loss_fake_avg = discriminator_loss_fake_acc / (
        sequential_batch_size * n_critic
    )

    # sys.stdout.write("\b\b")
    # sys.stdout.flush()

    return (
        generator_loss_fool_avg,
        generator_loss_sane_avg,
        discriminator_loss_real_avg,
        discriminator_loss_fake_avg,
    )


def main():
    # enable_log_layers()

    songs = load_audio_clips("input_sound/*.flac")

    # features_lengths_kernel_sizes = [
    #     (512, 1, 65),
    #     (32, 64, 31),
    #     (32, 256, 127),
    #     (32, 1024, 127),
    #     (32, 8192, 127),
    #     (8, 16384, 127),
    #     (4, 65536, 127),
    #     (2, 65536, 1),
    # ]

    patch_size = 65536

    latent_features = 64

    generator = Generator(num_latent_features=latent_features).cuda()
    discriminator = Discriminator().cuda()

    lr = 1e-5

    generator_optimizer = torch.optim.Adam(
        # generator_optimizer = torch.optim.RMSprop(
        generator.parameters(),
        lr=lr,
        # betas=(0.5, 0.999),
    )

    discriminator_optimizer = torch.optim.Adam(
        # discriminator_optimizer = torch.optim.RMSprop(
        discriminator.parameters(),
        lr=lr,
        # betas=(0.5, 0.999)
    )

    # restore_module(generator, "models/generator_9618.dat")
    # restore_module(discriminator, "models/discriminator_9618.dat")
    # restore_module(generator_optimizer, "models/generator_optimizer_9618.dat")
    # restore_module(discriminator_optimizer, "models/discriminator_optimizer_9618.dat")
    generator.train()
    discriminator.train()

    plt.ion()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    ax_tl = axes[0, 0]
    ax_tm = axes[0, 1]
    ax_tr = axes[0, 2]
    ax_bl = axes[1, 0]
    ax_bm = axes[1, 1]
    ax_br = axes[1, 2]

    plot_interval = 512
    loss_interval = 32
    plot_quantiles = 8
    sounds_per_plot = 4

    generator_loss_fool_plotter = LossPlotter(
        loss_interval, (0.8, 0.0, 0.0), plot_quantiles
    )
    generator_loss_sane_plotter = LossPlotter(
        loss_interval, (0.0, 0.2, 0.8), plot_quantiles
    )
    discriminator_loss_real_plotter = LossPlotter(
        loss_interval, (0.0, 0.7, 0.0), plot_quantiles
    )
    discriminator_loss_fake_plotter = LossPlotter(
        loss_interval, (0.8, 0.0, 0.0), plot_quantiles
    )
    discriminator_loss_combined_plotter = LossPlotter(
        loss_interval, (0.8, 0.7, 0.0), plot_quantiles
    )

    def save_things(iteration):
        save_module(
            generator,
            f"models/generator_{iteration + 1}.dat",
        )
        save_module(
            discriminator,
            f"models/discriminator_{iteration + 1}.dat",
        )
        save_module(
            generator_optimizer,
            f"models/generator_optimizer_{iteration + 1}.dat",
        )
        save_module(
            discriminator_optimizer,
            f"models/discriminator_optimizer_{iteration + 1}.dat",
        )

    current_iteration = 0
    try:
        for current_iteration in range(1_000_000_000):
            (
                generator_loss_fool,
                generator_loss_sane,
                discriminator_loss_real,
                discriminator_loss_fake,
            ) = train(
                all_audio_clips=songs,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                latent_features=latent_features,
                patch_size=patch_size,
            )
            generator_loss_fool_plotter.append(generator_loss_fool)
            generator_loss_sane_plotter.append(generator_loss_sane)
            discriminator_loss_fake_plotter.append(discriminator_loss_fake)
            discriminator_loss_real_plotter.append(discriminator_loss_real)
            discriminator_loss_combined_plotter.append(
                discriminator_loss_real + discriminator_loss_fake
            )

            progress_bar(
                current_iteration % plot_interval,
                plot_interval,
                f"total: {current_iteration + 1}",
            )

            time_to_plot = (
                (current_iteration + 1) % plot_interval
            ) == 0 or current_iteration <= 1

            if time_to_plot:
                with torch.no_grad():
                    generator.eval()
                    output_clips = []
                    for _ in range(sounds_per_plot):
                        generated_clip = generator(
                            random_initial_vector(
                                batch_size=1, num_features=latent_features
                            )
                        )
                        generated_clip = generated_clip.squeeze().detach().cpu()
                        output_clips.append(generated_clip)
                    generator.train()

                audio_clip, song_name = random.choice(songs)

                ax_tl.cla()
                ax_tm.cla()
                ax_tr.cla()
                ax_bl.cla()
                ax_bm.cla()
                ax_br.cla()

                ax_tl.title.set_text(f'Real Waveform: "{song_name}"')
                ax_tl.scatter(range(audio_clip.shape[1]), audio_clip[0].numpy(), s=1.0)
                ax_tl.scatter(range(audio_clip.shape[1]), audio_clip[1].numpy(), s=1.0)
                ax_tl.set_ylim(-1.0, 1.0)

                clip_to_plot = output_clips[0]

                ax_bl.title.set_text(f"Fake Audio Waveform")
                ax_bl.scatter(
                    range(clip_to_plot.shape[1]),
                    clip_to_plot[0].numpy(),
                    s=1.0,
                )
                ax_bl.scatter(
                    range(clip_to_plot.shape[1]),
                    clip_to_plot[1].numpy(),
                    s=1.0,
                )
                ax_bl.set_ylim(-1.0, 1.0)

                ax_tm.title.set_text(f'Real Spectrogram - "{song_name}"')
                ax_tm.imshow(
                    make_spectrogram(audio_clip[:2]).unsqueeze(2).repeat(1, 1, 3)
                )

                rgb_spectrogram = torch.stack(
                    [make_spectrogram(c) for c in output_clips[:3]], dim=2
                )
                ax_bm.title.set_text(f"Fake Audio Spectrograms (3x RGB Overlay)")
                ax_bm.imshow(rgb_spectrogram)

                ax_tr.title.set_text("Generator Score")
                generator_loss_fool_plotter.plot_to(ax_tr)
                generator_loss_sane_plotter.plot_to(ax_tr)
                # ax_tr.set_xlim(-1, current_iteration + 1)
                # ax_tr.set_yscale("log")

                ax_br.title.set_text("Discriminator Scores")
                discriminator_loss_fake_plotter.plot_to(ax_br)
                discriminator_loss_real_plotter.plot_to(ax_br)
                discriminator_loss_combined_plotter.plot_to(ax_br)
                # ax_br.set_xlim(-1, current_iteration + 1)
                # ax_br.set_yscale("log")

                fig.canvas.draw()
                fig.canvas.flush_events()

                plt_screenshot(fig).save(f"images/image_{current_iteration + 1}.png")

                for i_clip, clip in enumerate(output_clips):
                    clip_to_play = clip[:2]
                    clip_to_play = clip_to_play.detach()
                    mean_amp = torch.mean(clip_to_play, dim=1, keepdim=True)
                    clip_to_play -= mean_amp
                    max_amp = torch.max(torch.abs(clip_to_play), dim=1, keepdim=True)[0]
                    clip_to_play = 0.5 * clip_to_play / max_amp
                    torchaudio.save(
                        filepath=f"output_sound/output_{current_iteration + 1}_v{i_clip + 1}.flac",
                        src=clip_to_play.repeat(1, 2),
                        sample_rate=32000,
                    )
            if ((current_iteration + 1) % 65536) == 0:
                save_things(current_iteration)

    except KeyboardInterrupt as e:
        print("\n\nControl-C detected, saving model...\n")
        save_things(current_iteration)
        print("Exiting")
        exit(1)


if __name__ == "__main__":
    main()
