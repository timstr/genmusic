from operator import ge
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import random
import sounddevice
import time


def cosine_window(size, device):
    assert isinstance(size, int)
    ls = torch.tensor(np.linspace(0.0, 1.0, num=size, endpoint=False), device=device)
    assert ls.shape == (size,)
    return 0.5 - 0.5 * torch.cos(ls * 2.0 * np.pi)


def slice_along_dim(num_dims, dim, start=None, stop=None, step=None):
    assert isinstance(num_dims, int)
    begin = (slice(None),) * dim
    middle = (slice(start, stop, step),)
    end = (slice(None),) * (num_dims - dim - 1)
    ret = begin + middle + end
    assert len(ret) == num_dims
    return ret


def upsample_2x(x, dim):
    assert isinstance(x, torch.Tensor)
    assert isinstance(dim, int)
    num_dims = len(x.shape)
    assert dim < num_dims
    N = x.shape[dim]
    new_shape = list(x.shape)
    new_shape[dim] = 2 * N
    new_shape = tuple(new_shape)
    out = torch.zeros(new_shape, dtype=x.dtype, device=x.device)
    out[slice_along_dim(num_dims, dim, start=0, step=2)] = x
    x_shift = torch.cat(
        (
            x[slice_along_dim(num_dims, dim, start=1)],
            x[slice_along_dim(num_dims, dim, start=-1)],
        ),
        dim=dim,
    )
    assert x_shift.shape == x.shape
    out[slice_along_dim(num_dims, dim, start=1, step=2)] = 0.5 * (x + x_shift)
    return out


def downsample_2x(x, dim):
    assert isinstance(x, torch.Tensor)
    assert isinstance(dim, int)
    num_dims = len(x.shape)
    assert dim < num_dims
    evens = x[slice_along_dim(num_dims, dim, start=0, step=2)]
    odds = x[slice_along_dim(num_dims, dim, start=1, step=2)]
    return 0.5 * (evens + odds)


class RefineAudio(nn.Module):
    def __init__(self, sample_count, kernel_size, num_features, hidden_features):
        super(RefineAudio, self).__init__()
        assert isinstance(sample_count, int)
        assert isinstance(kernel_size, int)
        assert isinstance(num_features, int)
        assert isinstance(hidden_features, int)
        self.sample_count = sample_count
        self.num_features = num_features
        self.hidden_features = hidden_features

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        x0 = x
        assert x0.shape == (B, self.num_features, self.sample_count)
        x1 = self.conv1(x0)
        assert x1.shape == (B, self.hidden_features, self.sample_count)
        x2 = torch.relu(x1)
        x3 = self.conv2(x2)
        assert x3.shape == (B, self.num_features, self.sample_count)
        return x0 + x3


class DiscriminateAudio(nn.Module):
    def __init__(self, sample_count, kernel_size, audio_channels, hidden_features):
        super(DiscriminateAudio, self).__init__()
        assert isinstance(sample_count, int)
        assert isinstance(kernel_size, int)
        assert isinstance(audio_channels, int)
        assert isinstance(hidden_features, int)

        self.sample_count = sample_count
        self.audio_channels = audio_channels
        self.num_features = hidden_features

        self.conv1 = nn.Conv1d(
            in_channels=audio_channels,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )
        self.fc = nn.Linear(
            in_features=(4 * hidden_features), out_features=1  # min, max, mean, stddev
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        x0 = x[:, : self.audio_channels]
        assert x0.shape == (B, self.audio_channels, self.sample_count)
        x1 = self.conv1(x0)
        assert x1.shape == (B, self.num_features, self.sample_count // 2)
        x2 = torch.relu(x1)
        x3 = self.conv2(x2)
        assert x3.shape == (B, self.num_features, self.sample_count // 4)
        x4 = torch.relu(x3)
        x5_0 = torch.min(x4, dim=2)[0]
        x5_1 = torch.max(x4, dim=2)[0]
        x5_2 = torch.mean(x4, dim=2)
        x5_3 = torch.std(x4, dim=2)
        assert x5_0.shape == (B, self.num_features)
        assert x5_1.shape == (B, self.num_features)
        assert x5_2.shape == (B, self.num_features)
        assert x5_3.shape == (B, self.num_features)
        x5 = torch.cat((x5_0, x5_1, x5_2, x5_3), dim=1)
        assert x5.shape == (B, (4 * self.num_features))
        x6 = self.fc(x5)
        assert x6.shape == (B, 1)
        return x6

class GeneratorsAndDiscriminators(nn.Module):
    def __init__(self, generators, discriminators):
        super(GeneratorsAndDiscriminators, self).__init__()
        self.generators = nn.ModuleList(generators)
        self.discriminators = nn.ModuleList(discriminators)

def make_networks(layer_size, patch_size, num_features, device):
    assert isinstance(layer_size, int)
    assert isinstance(patch_size, int)
    assert isinstance(num_features, int)
    num_levels = (patch_size // layer_size).bit_length()

    generators = [
        RefineAudio(
            sample_count=layer_size,
            num_features=num_features,
            kernel_size=3,
            hidden_features=(2 * num_features),
        ).to(device)
        for _ in range(num_levels)
    ]

    discriminators = [
        DiscriminateAudio(
            sample_count=layer_size,
            kernel_size=3,
            audio_channels=2,
            hidden_features=num_features,
        ).to(device)
        for _ in range(num_levels)
    ]

    return GeneratorsAndDiscriminators(generators, discriminators), num_levels


def make_optimizers(networks, lr):
    assert isinstance(networks, list)
    assert isinstance(lr, float)
    optimizers = [torch.optim.Adam(network.parameters(), lr=lr) for network in networks]
    return optimizers


def random_initial_clip(num_features, layer_size, device):
    return 0.1 * torch.randn((num_features, layer_size), device=device)


def downsample_many_times(audio_clip, layer_size):
    assert isinstance(audio_clip, torch.Tensor)
    F, N = audio_clip.shape
    ret = []
    while audio_clip.shape[1] >= layer_size:
        N = audio_clip.shape[1]
        reshaped = audio_clip.reshape(F, N // layer_size, layer_size).permute(1, 0, 2)
        assert reshaped.shape == (N // layer_size, F, layer_size)
        ret.append(reshaped)
        audio_clip = downsample_2x(audio_clip, dim=1)
    ret.reverse()
    return ret


def upsample_and_generate_audio(initial_clip, generators, layer_size):
    # TODO: initial clip contains random noise
    # pass initial clip through generators one-by-one
    # use batch dimension to apply generators in parallel on neighbouring slices
    # return output batches at all dimensions (needed to train discriminators)
    assert initial_clip.shape[1:] == (layer_size,)
    clip = initial_clip
    ret = []
    for g in generators:
        F, N = clip.shape
        clips_batch = clip.reshape(F, N // layer_size, layer_size).permute(1, 0, 2)
        assert clips_batch.shape == (N // layer_size, F, layer_size)
        clips_generated_batch = g(clips_batch)
        F = clips_generated_batch.shape[1]
        assert clips_generated_batch.shape == (N // layer_size, F, layer_size)
        ret.append(clips_generated_batch)
        clip = clips_generated_batch.permute(1, 0, 2).reshape(F, N)
        clip = upsample_2x(clip, dim=1)
        clip = clip.detach()
        assert clip.shape == (F, 2 * N)
    return clip, ret


def train(
    audio_clip,
    layer_size,
    patch_size,
    generators,
    discriminators,
    optimizer,
    # generator_optimizers,
    # discriminator_optimizers,
    num_levels,
    num_features,
):
    assert isinstance(audio_clip, torch.Tensor)
    assert audio_clip.shape == (2, patch_size)
    assert isinstance(layer_size, int)
    assert isinstance(patch_size, int)
    # assert isinstance(generators, list)
    # assert isinstance(discriminators, list)
    # assert isinstance(generator_optimizers, list)
    # assert isinstance(discriminator_optimizers, list)
    assert isinstance(num_levels, int)
    assert isinstance(num_features, int)

    real_sample_batches = downsample_many_times(
        audio_clip=audio_clip, layer_size=layer_size
    )
    assert len(real_sample_batches) == num_levels

    initial_clip = random_initial_clip(
        num_features=num_features, layer_size=layer_size, device=audio_clip.device
    )

    final_clip, fake_sample_batches = upsample_and_generate_audio(
        initial_clip=initial_clip, generators=generators, layer_size=layer_size
    )
    assert len(fake_sample_batches) == num_levels

    # print(f"real_sample_batches:")
    # for i, b in enumerate(real_sample_batches):
    #     print(f"    {i} : {b.shape}")
    # print(f"fake_sample_batches:")
    # for i, b in enumerate(fake_sample_batches):
    #     print(f"    {i} : {b.shape}")

    # assert all(
    #     [
    #         (real.shape == fake.shape)
    #         for real, fake in zip(real_sample_batches, fake_sample_batches)
    #     ]
    # )

    discriminator_real_predictions = [
        d(real) for d, real in zip(discriminators, real_sample_batches)
    ]

    discriminator_fake_predictions = [
        d(fake) for d, fake in zip(discriminators, fake_sample_batches)
    ]

    generator_losses = []
    for pred in discriminator_fake_predictions:
        generator_losses.append(torch.mean((pred - 1.0) ** 2))

    discriminator_real_losses = []
    discriminator_fake_losses = []
    for pred in discriminator_real_predictions:
        discriminator_real_losses.append(torch.mean((pred - 1.0) ** 2))
    for pred in discriminator_fake_predictions:
        discriminator_fake_losses.append(torch.mean(pred ** 2))

    discriminator_losses = [
        real_loss + fake_loss
        for real_loss, fake_loss in zip(
            discriminator_real_losses, discriminator_fake_losses
        )
    ]

    total_loss = 0.0
    for g_loss in generator_losses:
        total_loss += g_loss
    for d_loss in discriminator_losses:
        total_loss += d_loss
    total_loss.backward()
    optimizer.step()

    # for g_opt, g_loss in zip(generator_optimizers, generator_losses):
    #     g_loss.backward(retain_graph=True)
    #     g_opt.step()

    # for d_opt, d_loss in zip(discriminator_optimizers, discriminator_losses):
    #     d_loss.backward(retain_graph=True)
    #     d_opt.step()

    return generator_losses, discriminator_losses, final_clip


def main():
    # cw = cosine_window(1024, "cpu")
    # plt.plot(cw)
    # plt.show()

    # N = 64
    # noise = torch.randn((3, 5, N, 12))
    # noise_upsampled = upsample_2x(noise, dim=2)
    # plt.plot(range(0, N * 2, 2), noise[0,0,:,0])
    # plt.plot(range(0, N * 2, 1), noise_upsampled[0,0,:,0])
    # plt.show()

    audio, sample_rate = torchaudio.load("sound/die taube auf dem dach 1.flac")
    audio_length = audio.shape[1]
    audio_twice = audio.repeat(1, 2)
    # print(f"audio.shape : {audio.shape}")
    # print(f"sample_rate : {sample_rate}")
    # plt.plot(audio[1])
    # plt.show()
    # sounddevice.play(audio.repeat(1, 10).permute(1, 0).numpy(), 32000)
    # time.sleep(10.0)

    layer_size = 256
    patch_size = 8192
    num_features = 8

    generators_and_discriminators, num_levels = make_networks(
        layer_size=layer_size,
        patch_size=patch_size,
        num_features=num_features,
        device="cuda",
    )
    lr = 0.001
    optimizer = torch.optim.Adam(generators_and_discriminators.parameters(), lr=lr)
    # generator_optimizers = make_optimizers(generators, lr=lr)
    # discriminator_optimizers = make_optimizers(discriminators, lr=lr)

    plt.ion()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    ax_tl = axes[0, 0]
    ax_tr = axes[0, 1]
    ax_bl = axes[1, 0]
    ax_br = axes[1, 1]

    all_generator_losses = [[] for _ in range(num_levels)]
    all_discriminator_losses = [[] for _ in range(num_levels)]

    for i in range(1_000_000):
        random_start_time = random.randrange(audio_length)
        audio_clip = audio_twice[:, random_start_time : random_start_time + patch_size]
        audio_clip = audio_clip.to("cuda")
        generator_losses, discriminator_losses, output_clip = train(
            audio_clip=audio_clip,
            layer_size=layer_size,
            patch_size=patch_size,
            generators=generators_and_discriminators.generators,
            discriminators=generators_and_discriminators.discriminators,
            optimizer=optimizer,
            # generator_optimizers=generator_optimizers,
            # discriminator_optimizers=discriminator_optimizers,
            num_levels=num_levels,
            num_features=num_features,
        )

        for i, g_loss in enumerate(generator_losses):
            all_generator_losses[i].append(g_loss.item())

        for i, d_loss in enumerate(discriminator_losses):
            all_discriminator_losses[i].append(d_loss.item())

        time_to_plot = True

        if time_to_plot:
            ax_tl.cla()
            ax_tr.cla()
            ax_bl.cla()
            ax_br.cla()

            ax_tl.plot(audio_clip[0].cpu().numpy())
            ax_tl.plot(audio_clip[1].cpu().numpy())

            ax_bl.plot(output_clip[0].detach().cpu().numpy())
            ax_bl.plot(output_clip[1].detach().cpu().numpy())

            for losses in all_generator_losses:
                ax_tr.plot(losses)

            for losses in all_discriminator_losses:
                ax_br.plot(losses)

            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == "__main__":
    main()
