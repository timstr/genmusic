import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import random
import sounddevice
import time
import glob
import PIL
import scipy.io.wavfile as wf


def save_model(model, filename):
    print(f'Saving model to "{filename}"')
    torch.save(model.state_dict(), filename)


def restore_model(model, filename):
    print('Restoring model from "{}"'.format(filename))
    model.load_state_dict(torch.load(filename))
    model.eval()


def plt_screenshot(plt_figure):
    pil_img = PIL.Image.frombytes(
        "RGB", plt_figure.canvas.get_width_height(), plt_figure.canvas.tostring_rgb()
    )
    return pil_img


def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1) == 0) and (n != 0)


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
        assert kernel_size % 2 == 1
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
        self.bn1 = nn.BatchNorm1d(num_features=hidden_features)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
        )
        self.bn2 = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        x0 = x
        assert x0.shape == (B, self.num_features, self.sample_count)
        x1 = self.bn1(self.conv1(x0))
        assert x1.shape == (B, self.hidden_features, self.sample_count)
        x2 = torch.relu(x1)
        x3 = self.bn2(self.conv2(x2))
        assert x3.shape == (B, self.num_features, self.sample_count)
        x4 = torch.tanh(0.5 * x3)
        # return x0 + x3
        return x0 + x4
        # return x3
        # return x4


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
            padding=((kernel_size - 1) // 2),
        )
        self.bn1 = nn.BatchNorm1d(num_features=hidden_features)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_features,
            out_channels=hidden_features * 2,
            kernel_size=kernel_size,
            stride=2,
            padding=((kernel_size - 1) // 2),
        )
        self.bn2 = nn.BatchNorm1d(num_features=hidden_features * 2)
        self.fc1 = nn.Linear(
            in_features=(sample_count * hidden_features // 2),
            out_features=(sample_count * hidden_features // 16),
        )
        self.bn3 = nn.BatchNorm1d(num_features=(sample_count * hidden_features // 16))
        self.fc2 = nn.Linear(
            in_features=(sample_count * hidden_features // 16),
            out_features=1,
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        x0 = x[:, : self.audio_channels]
        assert x0.shape == (B, self.audio_channels, self.sample_count)
        x1 = self.bn1(self.conv1(x0))
        assert x1.shape == (B, self.num_features, self.sample_count // 2)
        x2 = torch.relu(x1)
        x3 = self.bn2(self.conv2(x2))
        assert x3.shape == (B, 2 * self.num_features, self.sample_count // 4)
        x4 = torch.relu(x3)
        x5 = x4.reshape(B, (self.sample_count * self.num_features // 2))
        x5 = x5 / (torch.sqrt(torch.sum(torch.square(x5), dim=1, keepdim=True)))
        x6 = self.fc1(x5)
        x6 = x6 / (torch.sqrt(torch.sum(torch.square(x6), dim=1, keepdim=True)))
        x7 = torch.relu(x6)
        x8 = self.fc2(x7)
        assert x8.shape == (B, 1)
        x9 = torch.sigmoid(x8)
        return x9


class GeneratorsAndDiscriminators(nn.Module):
    def __init__(self, generators, discriminators):
        super(GeneratorsAndDiscriminators, self).__init__()
        self.generators = nn.ModuleList(generators)
        self.discriminators = nn.ModuleList(discriminators)


def make_networks(layer_size, patch_size, num_features, device, kernel_size):
    assert isinstance(layer_size, int)
    assert is_power_of_2(layer_size)
    assert isinstance(patch_size, int)
    assert is_power_of_2(patch_size)
    assert isinstance(num_features, int)
    num_levels = (patch_size // layer_size).bit_length()

    curr_size = layer_size
    for _ in range(num_levels - 1):
        curr_size *= 2
    assert curr_size == patch_size

    generators = [
        RefineAudio(
            sample_count=layer_size,
            num_features=num_features,
            kernel_size=kernel_size,
            hidden_features=(2 * num_features),
        ).to(device)
        for _ in range(num_levels)
    ]

    discriminators = [
        DiscriminateAudio(
            sample_count=layer_size,
            kernel_size=kernel_size,
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
    return 1.0 - 2.0 * torch.rand((num_features, layer_size), device=device)


def downsample_many_times(audio_clip, layer_size):
    assert isinstance(audio_clip, torch.Tensor)
    F, N = audio_clip.shape
    samples = []
    while audio_clip.shape[1] >= layer_size:
        N = audio_clip.shape[1]
        reshaped = audio_clip.reshape(F, N // layer_size, layer_size).permute(1, 0, 2)
        assert reshaped.shape == (N // layer_size, F, layer_size)
        samples.append(reshaped)
        audio_clip = downsample_2x(audio_clip, dim=1)
    samples.reverse()
    return samples


def upsample_and_generate_audio(initial_clip, generators, layer_size):
    # TODO: initial clip contains random noise
    # pass initial clip through generators one-by-one
    # use batch dimension to apply generators in parallel on neighbouring slices
    # return output batches at all dimensions (needed to train discriminators)
    assert initial_clip.shape[1:] == (layer_size,)
    clip = initial_clip
    generated = []
    residuals = []
    for i, g in enumerate(generators):
        F, N = clip.shape
        clips_batch = clip.reshape(F, N // layer_size, layer_size).permute(1, 0, 2)
        assert clips_batch.shape == (N // layer_size, F, layer_size)
        clips_generated_batch = g(clips_batch)
        # F = clips_generated_batch.shape[1]
        assert clips_generated_batch.shape == (N // layer_size, F, layer_size)
        residual = clips_generated_batch[:, :2] - clips_batch[:, :2]
        generated.append(clips_generated_batch)
        residuals.append(residual)
        if i + 1 < len(generators):
            clip = clips_generated_batch.permute(1, 0, 2).reshape(F, N)
            clip = upsample_2x(clip, dim=1)
            # clip = clip.detach()
            assert clip.shape == (F, 2 * N)
    return clip, generated, residuals


def sane_audio_loss(audio):
    assert isinstance(audio, torch.Tensor)
    F, N = audio.shape
    mean = torch.mean(audio, dim=1, keepdim=True)
    zero_mean_audio = audio - mean
    mean_audio_amp = torch.mean(torch.abs(zero_mean_audio))
    return torch.mean(torch.abs(mean)) + mean_audio_amp


def train(
    audio_clip,
    initial_clip,
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
    assert isinstance(initial_clip, torch.Tensor)
    assert initial_clip.shape == (num_features, layer_size)
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

    initial_clip[:2].copy_(real_sample_batches[0][0])

    final_clip, fake_sample_batches, residuals = upsample_and_generate_audio(
        initial_clip=initial_clip, generators=generators, layer_size=layer_size
    )
    assert len(fake_sample_batches) == num_levels
    assert len(residuals) == num_levels

    # print(f"real_sample_batches:")
    # for i, b in enumerate(real_sample_batches):
    #     print(f"    {i} : {b.shape}")
    # print(f"fake_sample_batches:")
    # for i, b in enumerate(fake_sample_batches):
    #     print(f"    {i} : {b.shape}")
    # print(f"residuals:")
    # for i, b in enumerate(residuals):
    #     print(f"    {i} : {b.shape}")

    assert all(
        [
            (real.shape == fake[:, :2].shape and real.shape == res.shape)
            for real, fake, res in zip(
                real_sample_batches, fake_sample_batches, residuals
            )
        ]
    )

    discriminator_real_predictions = [
        d(real) for d, real in zip(discriminators, real_sample_batches)
    ]

    discriminator_fake_predictions = [
        d(fake) for d, fake in zip(discriminators, fake_sample_batches)
    ]

    generator_reconstruction_losses = []
    for real, fake, res in zip(real_sample_batches, fake_sample_batches, residuals):
        # print(f"real.shape : {real.shape}")
        # print(f"fake.shape : {fake.shape}")
        # actual_res = fake[:, :2] - real
        # generator_reconstruction_losses.append(torch.mean(torch.square(res - actual_res)))

        generator_reconstruction_losses.append(
            torch.mean(torch.square(real - fake[:, :2]))
        )

    # generator_fool_losses = []
    # for pred in discriminator_fake_predictions:
    #     generator_fool_losses.append(torch.mean(torch.square(pred - 1.0)))

    # discriminator_real_losses = []
    # discriminator_fake_losses = []
    # for pred in discriminator_real_predictions:
    #     discriminator_real_losses.append(torch.mean(torch.square(pred - 1.0)))
    # for pred in discriminator_fake_predictions:
    #     discriminator_fake_losses.append(torch.mean(torch.square(pred)))

    # discriminator_losses = [
    #     real_loss + fake_loss
    #     for real_loss, fake_loss in zip(
    #         discriminator_real_losses, discriminator_fake_losses
    #     )
    # ]

    total_loss = 0.0

    for i, g_reconst_loss in enumerate(generator_reconstruction_losses):
        total_loss += g_reconst_loss

    # for i, g_fool_loss in enumerate(generator_fool_losses):
    #     total_loss += g_fool_loss
    # for i, d_loss in enumerate(discriminator_losses):
    #     total_loss += d_loss

    # total_loss += torch.mean(torch.square(final_clip[:2] - audio_clip))

    # total_loss += 100.0 * sane_audio_loss(final_clip)

    total_loss.backward()
    optimizer.step()

    # for g_opt, g_loss in zip(generator_optimizers, generator_losses):
    #     g_loss.backward(retain_graph=True)
    #     g_opt.step()

    # for d_opt, d_loss in zip(discriminator_optimizers, discriminator_losses):
    #     d_loss.backward(retain_graph=True)
    #     d_opt.step()

    # return generator_fool_losses, discriminator_losses, final_clip
    return generator_reconstruction_losses, generator_reconstruction_losses, final_clip


def main():
    # cw = cosine_window(1024, "cpu")
    # plt.plot(cw)
    # plt.show()

    # N = 64
    # noise = torch.randn((3, 5, N, 12))
    # noise_upsampled = upsample_2x(noise, dim=2)
    # noise_downsampled = downsample_2x(noise, dim=2)
    # plt.plot(range(0, N * 2, 2), noise[0,0,:,0])
    # plt.plot(range(0, N * 2, 1), noise_upsampled[0,0,:,0])
    # plt.plot(range(0, N * 2, 4), noise_downsampled[0,0,:,0])
    # plt.show()

    songs = []
    for filename in sorted(glob.glob("sound/*.flac")):
        song_audio, sample_rate = torchaudio.load(filename)
        songs.append(song_audio)
    # audio, sample_rate = torchaudio.load("sound/die taube auf dem dach 1.flac")
    # audio, sample_rate = torchaudio.load("sound/the shark 1.flac")
    audio_length = songs[0].shape[1]
    # print(f"audio.shape : {audio.shape}")
    # print(f"sample_rate : {sample_rate}")
    # plt.plot(audio[1])
    # plt.show()
    # sounddevice.play(audio.repeat(1, 10).permute(1, 0).numpy(), 32000)
    # time.sleep(10.0)

    layer_size = 256
    # patch_size = 8192
    # patch_size = 16384
    # patch_size = 32768
    patch_size = 65536
    num_features = 32

    generators_and_discriminators, num_levels = make_networks(
        layer_size=layer_size,
        patch_size=patch_size,
        num_features=num_features,
        device="cuda",
        kernel_size=127,
    )
    lr = 1e-4
    optimizer = torch.optim.Adam(generators_and_discriminators.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(generators_and_discriminators.parameters(), lr=lr)
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
        random_song = random.choice(songs)
        random_song_twice = random_song.repeat(1, 2)
        random_start_time = random.randrange(audio_length)
        audio_clip = random_song_twice[
            :, random_start_time : random_start_time + patch_size
        ]
        initial_clip = random_initial_clip(
            num_features=num_features, layer_size=layer_size, device="cuda"
        )
        audio_clip = audio_clip.to("cuda")
        generator_losses, discriminator_losses, output_clip = train(
            audio_clip=audio_clip,
            initial_clip=initial_clip,
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
        assert output_clip[:2].shape == audio_clip.shape

        for i_g, g_loss in enumerate(generator_losses):
            all_generator_losses[i_g].append(g_loss.item())

        for i_d, d_loss in enumerate(discriminator_losses):
            all_discriminator_losses[i_d].append(d_loss.item())

        time_to_plot = i % 256 == 0

        if time_to_plot:
            ax_tl.cla()
            ax_tr.cla()
            ax_bl.cla()
            ax_br.cla()

            ax_tl.plot(audio_clip[0, ::32].cpu().numpy())
            ax_tl.plot(audio_clip[1, ::32].cpu().numpy())
            ax_tl.set_ylim(-1.0, 1.0)

            ax_bl.plot(output_clip[0, ::32].detach().cpu().numpy())
            ax_bl.plot(output_clip[1, ::32].detach().cpu().numpy())
            ax_bl.set_ylim(-1.0, 1.0)

            for i_g_loss, losses in enumerate(all_generator_losses):
                t = i_g_loss / (num_levels - 1)
                ax_tr.plot(losses, c=(0.0, 1.0 - t, t))
            ax_tr.set_yscale("log")

            for i_d_loss, losses in enumerate(all_discriminator_losses):
                t = i_d_loss / (num_levels - 1)
                ax_br.plot(losses, c=(0.0, 1.0 - t, t))
            ax_br.set_yscale("log")

            fig.canvas.draw()
            fig.canvas.flush_events()

            clip_to_play = output_clip[:2]
            clip_to_play = clip_to_play.detach()
            mean_amp = torch.mean(clip_to_play, dim=1, keepdim=True)
            clip_to_play -= mean_amp
            max_amp = torch.max(torch.abs(clip_to_play), dim=1, keepdim=True)[0]
            clip_to_play = 0.1 * clip_to_play / max_amp
            # sounddevice.play(
            #     clip_to_play.repeat(1, 4).permute(1, 0).cpu().numpy(), 32000
            # )

            plt_screenshot(fig).save(f"images/image_{i}.png")

            save_model(generators_and_discriminators, f"models/model_{i}.dat")

            wf.write(
                f"output_sound/output_{i}.wav",
                32000,
                clip_to_play.repeat(1, 4).permute(1, 0).cpu().numpy(),
            )


if __name__ == "__main__":
    main()
