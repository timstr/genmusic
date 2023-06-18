import os

from loss_plotter import LossPlotter

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import random

from signal_processing import make_spectrogram
from audio_util import load_audio_clips, random_audio_batch, sane_audio_loss
from torch_utils import (
    CheckShape,
    Resample1d,
    Reshape,
    Log,
    ResidualAdd1d,
    WithNoise1d,
    WithNoise2d,
    enable_log_layers,
    save_module,
    restore_module,
)
from util import assert_eq, plt_screenshot
from progress_bar import progress_bar


class Generator(nn.Module):
    def __init__(self, num_latent_features):
        super(Generator, self).__init__()
        assert isinstance(num_latent_features, int)

        self.num_latent_features = num_latent_features

        self.temporal_features = 8
        self.frequency_features = 8
        self.fc_output_length = 33
        self.fc_output_features = 32
        self.fc_hidden_features = 128

        self.window_size = 128
        self.window = nn.parameter.Parameter(
            data=torch.hann_window(self.window_size, periodic=True), requires_grad=False
        )

        self.frequencies = (self.window_size // 2) + 1
        self.spectral_features = self.frequencies * self.frequency_features

        # def residual_block(in_features, hidden_features, out_features, kernel_size):
        #     assert (kernel_size % 2) == 1
        #     return ResidualAdd1d(
        #         nn.Sequential(
        #             nn.Conv1d(
        #                 in_channels=in_features,
        #                 out_channels=hidden_features,
        #                 kernel_size=kernel_size,
        #                 stride=1,
        #                 padding=(kernel_size - 1) // 2,
        #                 padding_mode="circular",
        #             ),
        #             nn.BatchNorm1d(num_features=hidden_features),
        #             nn.LeakyReLU(0.2, True),
        #             WithNoise1d(num_features=hidden_features),
        #             nn.Conv1d(
        #                 in_channels=hidden_features,
        #                 out_channels=out_features,
        #                 kernel_size=kernel_size,
        #                 stride=1,
        #                 padding=(kernel_size - 1) // 2,
        #                 padding_mode="circular",
        #             ),
        #         )
        #     )

        def conv_up_2x(in_features, out_features, kernel_size, output_padding=0):
            assert (kernel_size % 2) == 1
            return nn.ConvTranspose1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=kernel_size,
                stride=2,
                output_padding=output_padding,
                padding=(kernel_size - 1) // 2,
            )

        self.fully_connected = nn.Sequential(
            Log("generator fully connected 0"),
            CheckShape((self.num_latent_features,)),
            nn.Linear(
                in_features=self.num_latent_features,
                out_features=self.fc_hidden_features,
            ),
            # nn.BatchNorm1d(num_features=self.fc_hidden_features),
            nn.LeakyReLU(0.2, True),
            Log("generator fully connected 1"),
            nn.Linear(
                in_features=self.fc_hidden_features,
                out_features=(self.fc_output_features * self.fc_output_length),
            ),
            Log("generator fully connected 2"),
        )
        # print("Generator fully-connected:")
        # print(self.fully_connected)
        # print("")

        self.spectral_convs = nn.Sequential(
            CheckShape((self.fc_output_features, 33)),
            Log("generator spectral conv 0"),
            # Resample1d(new_length=65),
            # residual_block(
            #     in_features=self.fc_output_features,
            #     hidden_features=self.spectral_features,
            #     out_features=self.spectral_features,
            #     kernel_size=5,
            # ),
            conv_up_2x(
                in_features=self.fc_output_features,
                out_features=self.spectral_features,
                kernel_size=5,
            ),
            nn.LeakyReLU(0.2, True),
            WithNoise1d(num_features=self.spectral_features),
            CheckShape((self.spectral_features, 65)),
            Log("generator spectral conv 1"),
            # Resample1d(new_length=129),
            # residual_block(
            #     in_features=self.spectral_features,
            #     hidden_features=self.spectral_features,
            #     out_features=self.spectral_features,
            #     kernel_size=5,
            # ),
            conv_up_2x(
                in_features=self.spectral_features,
                out_features=self.spectral_features,
                kernel_size=5,
            ),
            CheckShape((self.spectral_features, 129)),
            Log("generator spectral conv 2"),
        )
        # print("Generator spectral convolutions:")
        # print(self.spectral_convs)
        # print("")

        self.temporal_convs = nn.Sequential(
            CheckShape((self.frequency_features // 2, 8192)),
            Log("generator temporal conv 0"),
            # Resample1d(new_length=16384),
            # residual_block(
            #     in_features=self.frequency_features // 2,
            #     hidden_features=self.temporal_features,
            #     out_features=self.temporal_features,
            #     kernel_size=5,
            # ),
            conv_up_2x(
                in_features=self.frequency_features // 2,
                out_features=self.temporal_features,
                kernel_size=31,
                output_padding=1,
            ),
            nn.LeakyReLU(0.2, True),
            WithNoise1d(num_features=self.temporal_features),
            CheckShape((self.temporal_features, 16384)),
            Log("generator temporal conv 1"),
            # Resample1d(new_length=32768),
            # residual_block(
            #     in_features=self.temporal_features,
            #     hidden_features=self.temporal_features,
            #     out_features=self.temporal_features,
            #     kernel_size=5,
            # ),
            conv_up_2x(
                in_features=self.temporal_features,
                out_features=self.temporal_features,
                kernel_size=31,
                output_padding=1,
            ),
            CheckShape((self.temporal_features, 32768)),
            nn.LeakyReLU(0.2, True),
            WithNoise1d(num_features=self.temporal_features),
            Log("generator temporal conv 2"),
            # Resample1d(new_length=65536),
            # residual_block(
            #     in_features=self.temporal_features,
            #     hidden_features=self.temporal_features,
            #     out_features=2,
            #     kernel_size=5,
            # ),
            conv_up_2x(
                in_features=self.temporal_features,
                out_features=2,
                kernel_size=5,
                output_padding=1,
            ),
            CheckShape((2, 65536)),
            Log("generator temporal conv 3"),
        )
        # print("Generator temporal convolutions")
        # print(self.temporal_convs)
        # print("")

    def forward(self, latent_codes):
        B, D = latent_codes.shape
        assert_eq(D, self.num_latent_features)

        x0 = latent_codes
        x1 = self.fully_connected(x0)

        assert_eq(x1.shape, (B, self.fc_output_features * self.fc_output_length))
        x2 = x1.reshape(B, self.fc_output_features, self.fc_output_length)
        x3 = self.spectral_convs(x2)
        assert_eq(x3.shape, (B, self.frequency_features * self.frequencies, 129))

        x4 = torch.complex(
            real=x3[:, : (self.frequency_features * self.frequencies // 2)],
            imag=x3[:, (self.frequency_features * self.frequencies // 2) :],
        )

        assert_eq(x4.shape, (B, self.frequency_features * self.frequencies // 2, 129))
        x5 = x4.reshape(B * self.frequency_features // 2, self.frequencies, 129)

        x6 = torch.istft(
            input=x5,
            n_fft=self.window_size,
            hop_length=(self.window_size // 2),
            window=self.window,
            center=True,
            return_complex=False,
            onesided=True,
        )
        assert_eq(x6.shape, (B * self.frequency_features // 2, 8192))

        x7 = x6.reshape(B, self.frequency_features // 2, 8192)
        x8 = self.temporal_convs(x7)
        assert_eq(x8.shape, (B, 2, 65536))

        return x8


class Discriminator(nn.Module):
    def __init__(self, num_latent_features):
        super(Discriminator, self).__init__()
        assert isinstance(num_latent_features, int)

        self.temporal_features = 8
        self.frequency_features = 8
        self.fc_input_length = 33
        self.fc_input_features = 32
        self.fc_hidden_features = 128
        self.num_latent_features = num_latent_features

        self.window_size = 128
        self.window = nn.parameter.Parameter(
            data=torch.hann_window(self.window_size, periodic=True), requires_grad=False
        )

        self.frequencies = (self.window_size // 2) + 1
        self.spectral_features = self.frequencies * self.frequency_features

        def conv_down_2x(in_features, out_features, kernel_size):
            assert (kernel_size % 2) == 1
            return nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
            )

        self.temporal_convs = nn.Sequential(
            Log("discriminator temporal conv 0"),
            conv_down_2x(
                in_features=2, out_features=self.temporal_features, kernel_size=5
            ),
            nn.BatchNorm1d(num_features=self.temporal_features),
            nn.LeakyReLU(0.2, True),
            Log("discriminator temporal conv 1"),
            conv_down_2x(
                in_features=self.temporal_features,
                out_features=self.temporal_features,
                kernel_size=31,
            ),
            nn.BatchNorm1d(num_features=self.temporal_features),
            nn.LeakyReLU(0.2, True),
            Log("discriminator temporal conv 2"),
            conv_down_2x(
                in_features=self.temporal_features,
                out_features=self.frequency_features // 2,
                kernel_size=31,
            ),
            nn.BatchNorm1d(num_features=self.frequency_features // 2),
            Log("discriminator temporal conv 3"),
        )
        # print("Discriminator temporal convolutions:")
        # print(self.temporal_convs)
        # print("")

        self.spectral_convs = nn.Sequential(
            Log("discriminator spectral conv 0"),
            conv_down_2x(
                in_features=self.spectral_features,
                out_features=self.spectral_features,
                kernel_size=5,
            ),
            nn.BatchNorm1d(num_features=self.spectral_features),
            nn.LeakyReLU(0.2, True),
            Log("discriminator spectral conv 1"),
            conv_down_2x(
                in_features=self.spectral_features,
                out_features=self.fc_input_features,
                kernel_size=5,
            ),
            nn.BatchNorm1d(num_features=self.fc_input_features),
            nn.LeakyReLU(0.2, True),
            Log("discriminator spectral conv 2"),
        )
        # print("Discriminator spectral convolutions:")
        # print(self.spectral_convs)
        # print("")

        self.fully_connected_latent = nn.Sequential(
            Log("discriminator fully connected latent 0"),
            CheckShape((self.fc_input_features * self.fc_input_length,)),
            nn.Linear(
                in_features=(self.fc_input_features * self.fc_input_length),
                out_features=self.fc_hidden_features,
            ),
            # nn.BatchNorm1d(num_features=self.fc_hidden_features),
            nn.LeakyReLU(0.2, True),
            Log("discriminator fully connected latent 1"),
            nn.Linear(
                in_features=self.fc_hidden_features,
                out_features=self.num_latent_features,
            ),
            Log("discriminator fully connected latent 2"),
        )
        # print("Discriminator fully-connected:")
        # print(self.fully_connected)
        # print("")

        self.fully_connected_score = nn.Sequential(
            Log("discriminator fully connected score 0"),
            nn.Linear(in_features=(self.num_latent_features), out_features=1,),
            Log("discriminator fully connected score 1"),
        )

    def forward(self, audio_clips):
        B, C, N = audio_clips.shape
        assert_eq(C, 2)
        assert_eq(N, 65536)

        x0 = audio_clips
        x1 = self.temporal_convs(x0)
        assert_eq(x1.shape, (B, self.frequency_features // 2, 8192))
        x2 = x1.reshape(B * self.frequency_features // 2, 8192)
        x3 = torch.stft(
            input=x2,
            n_fft=self.window_size,
            hop_length=(self.window_size // 2),
            window=self.window,
            center=True,
            pad_mode="circular",
            return_complex=True,
            onesided=True,
        )
        assert_eq(x3.shape, (B * self.frequency_features // 2, self.frequencies, 129))
        x4 = x3.reshape(B, self.frequency_features // 2, self.frequencies, 129)
        x5 = torch.cat([torch.real(x4), torch.imag(x4)], dim=1)
        assert_eq(x5.shape, (B, self.frequency_features, self.frequencies, 129))

        x6 = x5.reshape(B, self.frequency_features * self.frequencies, 129)

        x7 = self.spectral_convs(x6)
        assert_eq(x7.shape, (B, self.fc_input_features, self.fc_input_length))
        x8 = x7.reshape(B, self.fc_input_features * self.fc_input_length)

        latent_features = self.fully_connected_latent(x8)
        assert_eq(latent_features.shape, (B, self.num_latent_features))

        score = self.fully_connected_score(latent_features)
        assert_eq(score.shape, (B, 1))

        return latent_features, score


# HACK to test networks
# enable_log_layers()
# d = Discriminator(num_latent_features=64).cuda()
# s = d(torch.rand((1, 2, 65536), device="cuda"))
# g = Generator(num_latent_features=64).cuda()
# a = g(torch.rand((1, 64), device="cuda"))
# exit(-1)


def random_initial_vector(batch_size, num_features):
    return torch.randn((batch_size, num_features), dtype=torch.float32, device="cuda")
    # return -1.0 + 2.0 * torch.rand(
    #     (batch_size, num_features, num_samples), dtype=torch.float32, device="cuda"
    # )


# adapted from https://github.com/Lornatang/WassersteinGAN_GP-PyTorch/blob/f2e2659089a4fe4cb7e1c4edeb5c5b9912e9c348/wgangp_pytorch/utils.py#L39
def calculate_gradient_penalty(model, real_images, fake_images):
    device = real_images.device
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(
        True
    )

    features, model_interpolates = model(interpolates)
    grad_outputs = torch.ones(
        model_interpolates.size(), device=device, requires_grad=False
    )

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


def train(
    all_audio_clips,
    generator,
    discriminator,
    generator_optimizers,
    discriminator_optimizers,
    latent_features,
    patch_size,
):
    assert isinstance(all_audio_clips, list)
    assert isinstance(generator, nn.Module)
    assert isinstance(discriminator, nn.Module)
    assert isinstance(generator_optimizers, list)
    assert all([isinstance(o, torch.optim.Optimizer) for o in generator_optimizers])
    assert isinstance(discriminator_optimizers, list)
    assert all([isinstance(o, torch.optim.Optimizer) for o in discriminator_optimizers])
    assert isinstance(latent_features, int)
    assert isinstance(patch_size, int)

    parallel_batch_size = 2
    sequential_batch_size = 1

    n_critic = 5
    n_generator = 1
    n_all = n_critic + n_generator

    latent_match_acc = 0.0
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

        training_generator = mode >= n_critic

        with torch.no_grad():
            real_clips_batch = random_audio_batch(
                batch_size=parallel_batch_size,
                patch_size=patch_size,
                audio_clips=all_audio_clips,
            )

        initial_vector = random_initial_vector(
            batch_size=parallel_batch_size, num_features=latent_features,
        )

        fake_clips_batch = generator(initial_vector)

        # Assign scores to real and fake audio
        discriminator_real_features, discriminator_real_scores = discriminator(
            real_clips_batch
        )
        discriminator_fake_features, discriminator_fake_scores = discriminator(
            fake_clips_batch
        )

        total_loss = 0.0

        if training_generator:
            # generator wants to maximize the fake score
            l_fool = -torch.mean(discriminator_fake_scores)
            l_sane = sane_audio_loss(fake_clips_batch)
            total_loss += l_fool
            total_loss += l_sane
            model = generator
            optimizers = generator_optimizers
            generator_loss_fool_acc -= l_fool.detach().cpu().item()
            generator_loss_sane_acc += l_sane.detach().cpu().item()
        else:
            # discriminator wants to maximize the real score and minimize the fake score
            l_real = -torch.mean(discriminator_real_scores)
            l_fake = torch.mean(discriminator_fake_scores)
            l = l_real + l_fake
            total_loss += l
            model = discriminator
            optimizers = discriminator_optimizers
            discriminator_loss_real_acc -= l_real.detach().cpu().item()
            discriminator_loss_fake_acc += l_fake.detach().cpu().item()

            gradient_penalty = calculate_gradient_penalty(
                discriminator, real_clips_batch, fake_clips_batch
            )

            total_loss += gradient_penalty

            # discriminator latent features should match input latent features
            # l_latent = torch.mean((discriminator_fake_features - initial_vector) ** 2)
            # total_loss += l_latent
            # latent_match_acc += l_latent.detach().cpu().item()

            # also want all scores to not be arbitrarily large
            # total_loss += 0.1 * (torch.mean(discriminator_real_scores**2) + torch.mean(discriminator_fake_scores**2))

        if first_step:
            for o in optimizers:
                o.zero_grad()

        total_loss.backward()

        if last_step:
            if sequential_batch_size > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad[...] /= sequential_batch_size
            for o in optimizers:
                o.step()

        # if last_step and not training_generator:
        #     parameter_limit = 0.05
        #     with torch.no_grad():
        #         for p in discriminator.parameters():
        #             p.clamp_(min=-parameter_limit, max=parameter_limit)

        # console_animation_characters = "-\\|/"
        # sys.stdout.write("\b")
        # if i_batch + 1 < (sequential_batch_size * n_all):
        #     sys.stdout.write(
        #         console_animation_characters[
        #             i_batch % len(console_animation_characters)
        #         ]
        #     )
        # sys.stdout.flush()

    latent_match_avg = latent_match_acc / sequential_batch_size
    generator_loss_fool_avg = generator_loss_fool_acc / (
        sequential_batch_size * max(n_generator, 1)
    )
    generator_loss_sane_avg = generator_loss_sane_acc / (
        sequential_batch_size * max(n_generator, 1)
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
        latent_match_avg,
        generator_loss_fool_avg,
        generator_loss_sane_avg,
        discriminator_loss_real_avg,
        discriminator_loss_fake_avg,
    )


def main():
    # enable_log_layers()

    songs = load_audio_clips("input_sound/*.flac")

    patch_size = 65536

    latent_features = 32

    generator = Generator(num_latent_features=latent_features).cuda()
    discriminator = Discriminator(num_latent_features=latent_features).cuda()

    num_generator_params = sum(p.numel() for p in generator.parameters())
    num_discriminator_params = sum(p.numel() for p in discriminator.parameters())

    print(f"The generator has {num_generator_params} parameters")
    print(f"The discriminator has {num_discriminator_params} parameters")

    lr = 0.00001

    generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)

    discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    # last_iteration = 113850
    last_iteration = None

    if last_iteration is not None:
        restore_module(generator, f"models/generator_{last_iteration}.dat")
        restore_module(discriminator, f"models/discriminator_{last_iteration}.dat")
        restore_module(
            generator_optimizer, f"models/generator_optimizer_{last_iteration}.dat",
        )
        restore_module(
            discriminator_optimizer,
            f"models/discriminator_optimizer_{last_iteration}.dat",
        )
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

    latent_match_plotter = LossPlotter(loss_interval, (0.0, 0.0, 0.8), plot_quantiles)
    generator_loss_fool_plotter = LossPlotter(
        loss_interval, (0.8, 0.0, 0.0), plot_quantiles
    )
    # generator_loss_sane_plotter = LossPlotter(
    #     loss_interval, (0.0, 0.2, 0.8), plot_quantiles
    # )
    discriminator_loss_real_plotter = LossPlotter(
        loss_interval, (0.0, 0.7, 0.0), plot_quantiles
    )
    discriminator_loss_fake_plotter = LossPlotter(
        loss_interval, (0.0, 0.4, 0.0), plot_quantiles
    )
    # discriminator_loss_combined_plotter = LossPlotter(
    #     loss_interval, (0.8, 0.7, 0.0), plot_quantiles
    # )

    def save_things(iteration):
        save_module(
            generator, f"models/generator_{iteration + 1}.dat",
        )
        save_module(
            discriminator, f"models/discriminator_{iteration + 1}.dat",
        )
        save_module(
            generator_optimizer, f"models/generator_optimizer_{iteration + 1}.dat",
        )
        save_module(
            discriminator_optimizer,
            f"models/discriminator_optimizer_{iteration + 1}.dat",
        )

    current_iteration = 0 if last_iteration is None else last_iteration
    try:
        for _ in range(1_000_000_000):
            (
                latent_match,
                generator_loss_fool,
                generator_loss_sane,
                discriminator_loss_real,
                discriminator_loss_fake,
            ) = train(
                all_audio_clips=songs,
                generator=generator,
                discriminator=discriminator,
                generator_optimizers=[generator_optimizer],
                discriminator_optimizers=[discriminator_optimizer],
                latent_features=latent_features,
                patch_size=patch_size,
            )
            latent_match_plotter.append(latent_match)
            generator_loss_fool_plotter.append(generator_loss_fool)
            # generator_loss_sane_plotter.append(generator_loss_sane)
            discriminator_loss_fake_plotter.append(discriminator_loss_fake)
            discriminator_loss_real_plotter.append(discriminator_loss_real)
            # discriminator_loss_combined_plotter.append(
            #     discriminator_loss_real + discriminator_loss_fake
            # )

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
                    range(clip_to_plot.shape[1]), clip_to_plot[0].numpy(), s=1.0,
                )
                ax_bl.scatter(
                    range(clip_to_plot.shape[1]), clip_to_plot[1].numpy(), s=1.0,
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
                latent_match_plotter.plot_to(ax_tr)
                generator_loss_fool_plotter.plot_to(ax_tr)
                # generator_loss_sane_plotter.plot_to(ax_tr)
                # ax_tr.set_xlim(-1, current_iteration + 1)
                # ax_tr.set_yscale("log")

                ax_br.title.set_text("Discriminator Scores")
                discriminator_loss_fake_plotter.plot_to(ax_br)
                discriminator_loss_real_plotter.plot_to(ax_br)
                # discriminator_loss_combined_plotter.plot_to(ax_br)
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

            current_iteration += 1

    except KeyboardInterrupt as e:
        print("\n\nControl-C detected, saving model...\n")
        save_things(current_iteration)
        print("Exiting")
        exit(1)


if __name__ == "__main__":
    main()
