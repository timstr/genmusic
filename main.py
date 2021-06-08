import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import torch
import torch.nn as nn
import torch.fft as fft
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import PIL
import scipy.io.wavfile as wf
import sys
import scipy.signal


def progress_bar(current, total, message):
    if not sys.stdout.isatty():
        return
    i = current + 1
    bar_fill = "=" * (i * 50 // total)
    sys.stdout.write("\r[%-50s] %d/%d" % (bar_fill, i, total))
    if (message is not None) and (len(message) > 0):
        sys.stdout.write(" ")
        sys.stdout.write(message)
    if i == total:
        sys.stdout.write("\n")
    sys.stdout.flush()


convolution_padding_frequency_domain = "zeros"
convolution_padding_time_domain = "circular"


def make_spectrogram(x, normalize=True):
    assert len(x.shape) == 2
    assert x.shape[1] > x.shape[0]
    sgs = []
    for i in range(x.shape[0]):
        xi = x[i]
        _, _, sg = scipy.signal.spectrogram(
            xi.detach().cpu().numpy(),
            window="blackmanharris",
            nperseg=256,
            noverlap=128,
        )
        sg = np.log(np.clip(np.abs(sg), a_min=1e-6, a_max=128))
        assert len(sg.shape) == 2
        sgs.append(torch.tensor(sg))
    spectrogram = torch.cat(sgs, dim=0).flip(0)
    if normalize:
        spectrogram_min = torch.min(spectrogram)
        spectrogram_max = torch.max(spectrogram)
        spectrogram = (spectrogram - spectrogram_min) / (
            spectrogram_max - spectrogram_min
        )
    return spectrogram


def save_module(the_module, filename):
    print(f'Saving module to "{filename}"')
    torch.save(the_module.state_dict(), filename)


def restore_module(the_module, filename):
    print('Restoring module from "{}"'.format(filename))
    the_module.load_state_dict(torch.load(filename))


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
    ls = torch.tensor(
        np.linspace(0.0, 1.0, num=size, endpoint=False),
        dtype=torch.float,
        device=device,
    )
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
    # return x.repeat_interleave(2, dim=dim)


# hidden_activation_function = torch.relu
# hidden_activation_function = torch.sin
# hidden_activation_function = torch.sigmoid
# hidden_activation_function_generator = torch.relu
hidden_activation_function_generator = torch.nn.LeakyReLU()
hidden_activation_function_discriminator = torch.nn.LeakyReLU()

fft_norm = "ortho"


def downsample_2x(x, dim):
    assert isinstance(x, torch.Tensor)
    assert isinstance(dim, int)
    num_dims = len(x.shape)
    assert dim < num_dims
    evens = x[slice_along_dim(num_dims, dim, start=0, step=2)]
    odds = x[slice_along_dim(num_dims, dim, start=1, step=2)]
    return 0.5 * (evens + odds)


def make_conv_same(in_channels, out_channels, kernel_size, padding_mode):
    assert isinstance(in_channels, int)
    assert isinstance(out_channels, int)
    assert isinstance(kernel_size, int)
    assert isinstance(padding_mode, str)
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=((kernel_size - 1) // 2),
        padding_mode=padding_mode,
    )


def make_conv_up(in_channels, out_channels, kernel_size, scale_factor):
    assert scale_factor <= kernel_size
    assert ((kernel_size - scale_factor) % 2) == 0
    return nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=scale_factor,
        padding=((kernel_size - scale_factor) // 2),
        output_padding=0,
    )


def make_conv_down(in_channels, out_channels, kernel_size):
    assert kernel_size % 2 == 1
    return (
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size // 2) - 1),
        ),
    )


def make_positional_encoding(num_channels, length, device):
    assert isinstance(num_channels, int)
    assert isinstance(length, int)
    t = torch.tensor(
        np.linspace(0.0, 1.0, num=length, endpoint=False),
        dtype=torch.float,
        device=device,
    )
    f = torch.tensor(
        np.linspace(1.0, 1.0 + num_channels, num=num_channels, endpoint=False),
        dtype=torch.float,
        device=device,
    )

    tf = t.unsqueeze(0) * f.unsqueeze(1)
    assert tf.shape == (num_channels, length)

    pos_enc = torch.sin(np.pi * tf)

    return pos_enc


class WithNoise1d(nn.Module):
    def __init__(self, num_features):
        super(WithNoise1d, self).__init__()
        assert isinstance(num_features, int)
        self.num_features = num_features
        self.weights = nn.parameter.Parameter(
            torch.ones((num_features,), dtype=torch.float)
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B, F, N = x.shape
        assert F == self.num_features
        noise = -1.0 + 2.0 * torch.rand((B, F, N), dtype=x.dtype, device=x.device)
        weight = self.weights.reshape(1, F, 1)
        return x + weight * noise


class ConcatenateConstant(nn.Module):
    def __init__(self, constant):
        super(ConcatenateConstant, self).__init__()
        assert isinstance(constant, torch.Tensor)
        assert len(constant.shape) == 2
        self.constant = constant

    def forward(self, x):
        assert x.shape[2:] == self.constant.shape[1:]
        B = x.shape[0]
        return torch.cat((x, self.constant.unsqueeze(0).repeat(B, 1, 1)), dim=1)


class Apply(nn.Module):
    def __init__(self, fn):
        super(Apply, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def make_many_convs(
    sizes,
    kernel_size,
    padding_mode,
    length,
    positional_features,
    device,
    activation_function,
):
    assert isinstance(sizes, tuple)
    assert all([isinstance(size, int) for size in sizes])
    assert isinstance(kernel_size, int)
    assert isinstance(padding_mode, str)
    assert isinstance(length, int)
    assert isinstance(positional_features, int)

    pos_enc = make_positional_encoding(positional_features, length, device)

    modules = []
    N = len(sizes) - 1
    for i in range(N):
        size_prev = sizes[i]
        size_next = sizes[i + 1]
        modules.append(ConcatenateConstant(pos_enc))
        modules.append(
            make_conv_same(
                in_channels=size_prev + positional_features,
                out_channels=size_next,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
            )
        )
        if i + 1 < N:
            modules.append(nn.BatchNorm1d(num_features=size_next))
            modules.append(Apply(activation_function))
    return nn.Sequential(*modules)


class RefineAudio(nn.Module):
    def __init__(
        self,
        sample_count,
        kernel_size,
        num_input_features,
        num_output_features,
        positional_features,
        hidden_features,
        num_hidden_layers,
        num_input_style_features,
        num_hidden_style_features,
    ):
        super(RefineAudio, self).__init__()
        assert isinstance(sample_count, int)
        assert isinstance(kernel_size, int)
        assert kernel_size % 2 == 1
        assert isinstance(num_input_features, int)
        assert isinstance(num_output_features, int)
        assert isinstance(hidden_features, int)
        assert isinstance(num_hidden_layers, int)
        assert isinstance(num_input_style_features, int)
        assert isinstance(num_hidden_style_features, int)
        self.sample_count = sample_count
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.num_input_style_features = num_input_style_features
        self.num_hidden_style_features = num_hidden_style_features
        self.hidden_features = hidden_features
        self.positional_features = positional_features

        self.temporal_pathway = make_many_convs(
            (
                num_input_features + num_hidden_style_features,
                *((hidden_features,) * num_hidden_layers),
                num_output_features,
            ),
            kernel_size=kernel_size,
            padding_mode=convolution_padding_time_domain,
            length=sample_count,
            positional_features=0,  # positional_features,
            device="cuda",
            activation_function=hidden_activation_function_generator,
        )
        self.frequency_pathway = make_many_convs(
            (
                # num_input_features * 2,
                (num_input_features + num_hidden_style_features) * 2,
                *((hidden_features * 2,) * num_hidden_layers),
                num_output_features * 2,
            ),
            kernel_size=kernel_size,
            padding_mode=convolution_padding_frequency_domain,
            length=(self.sample_count // 2 + 1),
            positional_features=positional_features,
            device="cuda",
            activation_function=hidden_activation_function_generator,
        )

        self.noise_scale = nn.parameter.Parameter(data=torch.tensor(0.1))

        self.style_transform = nn.Sequential(
            nn.Linear(
                in_features=(num_input_style_features),
                out_features=(sample_count * num_hidden_style_features),
            ),
        )

    def forward(self, clips, styles):
        assert isinstance(clips, torch.Tensor)
        assert isinstance(styles, torch.Tensor)

        B = clips.shape[0]

        assert clips.shape == (B, self.num_input_features, self.sample_count)
        assert styles.shape == (
            B,
            self.num_input_style_features,
        )

        styles_transformed = self.style_transform(styles)
        assert styles_transformed.shape == (
            B,
            self.num_hidden_style_features * self.sample_count,
        )
        styles_transformed = styles_transformed.reshape(
            B, self.num_hidden_style_features, self.sample_count
        )

        clip_and_style = torch.cat([clips, styles_transformed], dim=1)
        assert clip_and_style.shape == (
            B,
            self.num_input_features + self.num_hidden_style_features,
            self.sample_count,
        )
        # clip_and_style = clips

        x = clip_and_style
        x = clip_and_style + self.noise_scale * (
            -1.0 + 2.0 * torch.rand_like(clip_and_style)
        )
        # x = clip_and_style + 0.05 * (-1.0 + 2.0 * torch.rand_like(clip_and_style))

        x0 = x
        num_freqs = self.sample_count // 2 + 1
        assert x0.shape == (
            B,
            self.num_input_features + self.num_hidden_style_features,
            self.sample_count,
        )
        x1 = fft.rfft(x0, n=self.sample_count, dim=2, norm=fft_norm)
        assert x1.shape == (
            B,
            self.num_input_features + self.num_hidden_style_features,
            num_freqs,
        )
        # x1 = x1[:, :, :(self.sample_count // 2)]
        x1 = torch.cat([torch.real(x1), torch.imag(x1)], dim=1)
        assert x1.shape == (
            B,
            # self.num_input_features * 2,
            (self.num_input_features + self.num_hidden_style_features) * 2,
            num_freqs,
        )

        x2 = self.frequency_pathway(x1)
        assert x2.shape == (B, self.num_output_features * 2, num_freqs)

        x3 = torch.complex(
            real=x2[:, : self.num_output_features],
            imag=x2[:, self.num_output_features :],
        )
        assert x3.shape == (B, self.num_output_features, num_freqs)
        x4 = fft.irfft(x3, n=self.sample_count, dim=2, norm=fft_norm)
        assert x4.shape == (B, self.num_output_features, self.sample_count)

        x5 = self.temporal_pathway(x0)
        assert x5.shape == (B, self.num_output_features, self.sample_count)

        # return x4 + x5
        assert self.num_input_features >= self.num_output_features
        return clips[:, : self.num_output_features, :] + x4 + x5
        # return clips + x4 + x5
        # return clips + x5


class DiscriminateAudio(nn.Module):
    def __init__(
        self,
        sample_count,
        kernel_size,
        audio_channels,
        hidden_features,
        positional_features,
        num_hidden_layers,
    ):
        super(DiscriminateAudio, self).__init__()
        assert isinstance(sample_count, int)
        assert isinstance(kernel_size, int)
        assert isinstance(audio_channels, int)
        assert isinstance(hidden_features, int)
        assert isinstance(positional_features, int)
        assert isinstance(num_hidden_layers, int)

        self.sample_count = sample_count
        self.audio_channels = audio_channels
        self.hidden_features = hidden_features
        self.temporal_pathway = make_many_convs(
            (
                audio_channels,
                *((hidden_features,) * num_hidden_layers),
                hidden_features,
            ),
            kernel_size=kernel_size,
            padding_mode=convolution_padding_time_domain,
            length=sample_count,
            positional_features=0,  # positional_features,
            device="cuda",
            activation_function=hidden_activation_function_discriminator,
        )
        self.frequency_pathway = make_many_convs(
            (
                audio_channels * 2,
                *((hidden_features * 2,) * num_hidden_layers),
                hidden_features * 2,
            ),
            kernel_size=kernel_size,
            padding_mode=convolution_padding_frequency_domain,
            length=(sample_count // 2 + 1),
            positional_features=positional_features,
            device="cuda",
            activation_function=hidden_activation_function_discriminator,
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=hidden_features * 6,
                # in_features=hidden_features * 2,
                out_features=hidden_features * 2,
            ),
            Apply(hidden_activation_function_discriminator),
            nn.Linear(in_features=hidden_features * 2, out_features=1),
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        x0 = x
        num_freqs = self.sample_count // 2 + 1
        assert x0.shape == (B, self.audio_channels, self.sample_count)
        x1 = fft.rfft(x0, n=self.sample_count, dim=2, norm=fft_norm)
        assert x1.shape == (B, self.audio_channels, num_freqs)
        # x1 = x1[:, :, :(self.sample_count // 2)]
        x1 = torch.cat([torch.real(x1), torch.imag(x1)], dim=1)
        assert x1.shape == (B, self.audio_channels * 2, num_freqs)

        x2 = self.frequency_pathway(x1)
        assert x2.shape == (B, self.hidden_features * 2, num_freqs)

        x3 = self.temporal_pathway(x0)
        assert x3.shape == (B, self.hidden_features, self.sample_count)

        x4 = torch.mean(x2, dim=2)
        assert x4.shape == (B, self.hidden_features * 2)
        x5 = torch.logsumexp(x2, dim=2)
        assert x5.shape == (B, self.hidden_features * 2)
        x6 = torch.mean(x3, dim=2)
        assert x6.shape == (B, self.hidden_features)
        x7 = torch.logsumexp(x3, dim=2)
        assert x7.shape == (B, self.hidden_features)
        x8 = torch.cat((x4, x5, x6, x7), dim=1)
        assert x8.shape == (B, self.hidden_features * 6)
        # x8 = torch.cat((x6, x7), dim=1)
        # assert x8.shape == (B, self.hidden_features * 2)
        x9 = self.fc(x8)
        assert x9.shape == (B, 1)

        x12 = x9
        # x12 = torch.tanh(x9)
        # x12 = 0.5 + 0.5 * torch.tanh(x9)

        return x12


class GeneratorsAndDiscriminators(nn.Module):
    def __init__(self, generators, discriminators):
        super(GeneratorsAndDiscriminators, self).__init__()
        self.generators = nn.ModuleList(generators)
        self.discriminators = nn.ModuleList(discriminators)


def make_networks(
    layer_size,
    patch_size,
    upsample_factor,
    max_hidden_features,
    min_hidden_features,
    generator_hidden_layers,
    discriminator_hidden_layers,
    positional_features,
    device,
    kernel_size,
    num_input_style_features,
    num_hidden_style_features,
):
    assert isinstance(layer_size, int)
    assert is_power_of_2(layer_size)
    assert isinstance(patch_size, int)
    assert is_power_of_2(patch_size)
    assert isinstance(upsample_factor, int)
    assert is_power_of_2(upsample_factor)
    assert isinstance(max_hidden_features, int)
    assert isinstance(min_hidden_features, int)
    assert isinstance(generator_hidden_layers, int)
    assert isinstance(discriminator_hidden_layers, int)
    assert isinstance(positional_features, int)
    assert isinstance(kernel_size, int)
    assert isinstance(num_input_style_features, int)
    assert isinstance(num_hidden_style_features, int)

    num_levels = 1
    current_output_size = layer_size
    while current_output_size < patch_size:
        num_levels += 1
        current_output_size *= upsample_factor
    assert current_output_size == patch_size

    curr_size = layer_size
    for _ in range(num_levels - 1):
        curr_size *= upsample_factor
    assert curr_size == patch_size

    generators = []
    generator_hidden_features = max_hidden_features
    for _ in range(num_levels):
        next_generator_hidden_features = max(
            generator_hidden_features // 2, min_hidden_features
        )
        g = RefineAudio(
            sample_count=layer_size,
            num_input_features=generator_hidden_features,
            num_output_features=next_generator_hidden_features,
            kernel_size=kernel_size,
            hidden_features=generator_hidden_features,
            positional_features=positional_features,
            num_hidden_layers=generator_hidden_layers,
            num_input_style_features=num_input_style_features,
            num_hidden_style_features=num_hidden_style_features,
        ).to(device)
        generators.append(g)
        generator_hidden_features = next_generator_hidden_features

    discriminators = []
    discriminator_hidden_features = max_hidden_features
    for _ in range(num_levels):
        d = DiscriminateAudio(
            sample_count=layer_size,
            kernel_size=kernel_size,
            audio_channels=2,
            hidden_features=discriminator_hidden_features,
            positional_features=positional_features,
            num_hidden_layers=discriminator_hidden_layers,
        ).to(device)
        discriminators.append(d)
        discriminator_hidden_features = max(
            discriminator_hidden_features // 2, min_hidden_features
        )

    return GeneratorsAndDiscriminators(generators, discriminators), num_levels


# def make_optimizers(networks, lr):
#     assert isinstance(networks, list)
#     assert isinstance(lr, float)
#     optimizers = [torch.optim.Adam(network.parameters(), lr=lr) for network in networks]
#     return optimizers


# def random_initial_clip(num_features, layer_size, device):
#     return torch.rand((num_features, layer_size), device=device)


def random_training_batch(batch_size, patch_size, audio_clips):
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

    random_crop_offset = random.randrange(0, N - patch_size)
    clips_batch = clips_batch[
        :, :, random_crop_offset : random_crop_offset + patch_size
    ]

    return clips_batch


def downsample_many_times(clips_batch, layer_size, downsample_factor):
    assert isinstance(clips_batch, torch.Tensor)
    assert isinstance(layer_size, int)
    assert isinstance(downsample_factor, int)
    assert is_power_of_2(downsample_factor)
    batch_size, F, N = clips_batch.shape
    assert F == 2

    samples = []
    while clips_batch.shape[2] >= layer_size:
        N = clips_batch.shape[2]

        # if N == layer_size:
        #     random_crop_offset = 0
        # else:
        #     random_crop_offset = random.randrange(N - layer_size)
        #     # random_crop_offset = random.randrange(
        #     #     (N - (layer_size // 2)) // (layer_size // 2)
        #     # ) * (layer_size // 2)

        # clips_cropped = clips_batch[
        #     :, :, random_crop_offset : random_crop_offset + layer_size
        # ]
        # samples.append(clips_cropped)

        assert N % layer_size == 0
        clips_split = clips_batch.reshape(
            batch_size, 2, N // layer_size, layer_size
        ).permute(0, 2, 1, 3)
        assert clips_split.shape == (batch_size, N // layer_size, 2, layer_size)
        clips_split = clips_split.reshape(batch_size * N // layer_size, 2, layer_size)

        samples.append(clips_split)

        current_factor = 1
        while current_factor < downsample_factor:
            clips_batch = downsample_2x(clips_batch, dim=2)
            current_factor *= 2
        assert current_factor == downsample_factor

    samples.reverse()
    return samples


# def upsample_and_generate_training_batches(
#     batch_size,
#     layer_size,
#     # num_features,
#     # num_input_style_features,
#     generators,
# ):
#     assert isinstance(batch_size, int)
#     assert isinstance(layer_size, int)

#     assert len(generators) > 0
#     initial_generator = generators[0]
#     assert isinstance(initial_generator, RefineAudio)
#     initial_num_input_features = initial_generator.num_input_features

#     clips_batch = torch.rand((batch_size, initial_num_input_features, layer_size), device="cuda")
#     # clips_batch = torch.zeros((batch_size, num_features, layer_size), device="cuda")
#     # styles_batch = -1.0 + 2.0 * torch.rand(
#     #     (batch_size, num_input_style_features), device="cuda"
#     # )

#    = []

#     for i, g in enumerate(generators):
#         assert isinstance(g, RefineAudio)
#         assert clips_batch.shape == (batch_size, g.num_input_features, layer_size)
#         clips_refined = g(
#             clips=clips_batch,
#             # styles=styles_batch
#         )
#         assert clips_refined.shape == (batch_size, g.num_output_features, layer_size)

#         generated.append(clips_refined)

#         if i + 1 < len(generators):
#             clips_upsampled = upsample_2x(clips_refined, dim=2)
#             assert clips_upsampled.shape == (batch_size, g.num_output_features, 2 * layer_size)

#             clips_upsampled = torch.cat((clips_upsampled, clips_upsampled), dim=2)
#             # random_crop_offset = random.choice(
#             #     [0, layer_size // 2, layer_size, (layer_size * 3) // 2]
#             # )

#             random_crop_offset = random.randrange(layer_size)
#             # random_crop_offset = random.choice([0, layer_size])
#             clips_cropped = clips_upsampled[
#                 :, :, random_crop_offset : random_crop_offset + layer_size
#             ]
#             clips_batch = clips_cropped
#             assert clips_batch.shape == (batch_size, g.num_output_features, layer_size)

#     return generated


def overlap_add_split(clips, window_size):
    assert isinstance(clips, torch.Tensor)
    assert isinstance(window_size, int)

    B, F, N = clips.shape

    assert N % window_size == 0

    clips_in_phase_batch = clips.reshape(B, F, N // window_size, window_size).permute(
        0, 2, 1, 3
    )
    assert clips_in_phase_batch.shape == (B, N // window_size, F, window_size)

    # rotate clip to the left by half a window
    split_point_forward = window_size // 2
    clip_rotated = torch.cat(
        (clips[:, :, split_point_forward:], clips[:, :, :split_point_forward]), dim=2
    )
    assert clip_rotated.shape == (B, F, N)

    clips_out_of_phase_batch = clip_rotated.reshape(
        B, F, N // window_size, window_size
    ).permute(0, 2, 1, 3)
    assert clips_out_of_phase_batch.shape == (B, N // window_size, F, window_size)

    return clips_in_phase_batch, clips_out_of_phase_batch


def overlap_add_combine(in_phase, out_of_phase, window_size, window):
    assert isinstance(in_phase, torch.Tensor)
    assert isinstance(out_of_phase, torch.Tensor)
    assert isinstance(window_size, int)
    assert isinstance(window, torch.Tensor)
    assert window.shape == (window_size,)

    B, M, F, W = in_phase.shape
    assert W == window_size
    assert in_phase.shape == out_of_phase.shape
    N = M * window_size

    window = window.reshape(1, 1, 1, window_size)

    clips_generated_in_phase_batch = window * in_phase
    clips_generated_out_of_phase_batch = window * out_of_phase
    # rotate clip to the right by half a layer
    split_point_backward = N - (window_size // 2)
    clips_generated_in_phase = clips_generated_in_phase_batch.permute(
        0, 2, 1, 3
    ).reshape(B, F, N)
    clips_generated_out_of_phase_rotated = clips_generated_out_of_phase_batch.permute(
        0, 2, 1, 3
    ).reshape(B, F, N)
    clips_generated_out_of_phase = torch.cat(
        (
            clips_generated_out_of_phase_rotated[:, :, split_point_backward:],
            clips_generated_out_of_phase_rotated[:, :, :split_point_backward],
        ),
        dim=2,
    )
    assert clips_generated_out_of_phase.shape == (B, F, N)
    clips_generated_overlap_added = (
        clips_generated_in_phase + clips_generated_out_of_phase
    )
    return clips_generated_overlap_added


def generate_full_audio_clip_batch(
    batch_size,
    layer_size,
    upsample_factor,
    style_vectors,
    generators,
    device,
    overlap_and_add,
):
    assert isinstance(batch_size, int)
    assert isinstance(layer_size, int)
    assert isinstance(upsample_factor, int)
    assert len(generators) > 0
    initial_generator = generators[0]
    assert isinstance(initial_generator, RefineAudio)
    assert isinstance(overlap_and_add, bool)
    initial_num_input_features = initial_generator.num_input_features
    assert style_vectors.shape == (
        batch_size,
        initial_generator.num_input_style_features,
    )

    clip = torch.rand(
        (batch_size, initial_num_input_features, layer_size), device=device
    )
    # generated = []

    window = cosine_window(layer_size, device=device)

    for i, g in enumerate(generators):
        assert isinstance(g, RefineAudio)

        F_in = g.num_input_features
        F_out = g.num_output_features

        N = layer_size * upsample_factor ** i
        assert clip.shape == (batch_size, F_in, N)
        M = N // layer_size

        # Refine the signal in small slices, then overlap and add
        # clips_in_phase_batch = clip.reshape(F_in, N // layer_size, layer_size).permute(
        #     1, 0, 2
        # )
        # assert clips_in_phase_batch.shape == (N // layer_size, F_in, layer_size)
        # # rotate clip to the left by half a layer
        # split_point_forward = layer_size // 2
        # clip_rotated = torch.cat(
        #     (clip[:, split_point_forward:], clip[:, :split_point_forward]), dim=1
        # )
        # assert clip_rotated.shape == (F_in, N)
        # clips_out_of_phase_batch = clip_rotated.reshape(
        #     F_in, N // layer_size, layer_size
        # ).permute(1, 0, 2)
        # assert clips_out_of_phase_batch.shape == (N // layer_size, F_in, layer_size)

        assert style_vectors.shape == (batch_size, g.num_input_style_features)
        styles_repeated = style_vectors.repeat_interleave(repeats=M, dim=0)
        assert styles_repeated.shape == (batch_size * M, g.num_input_style_features)

        if overlap_and_add:
            clips_in_phase_batch, clips_out_of_phase_batch = overlap_add_split(
                clip, window_size=layer_size
            )

            assert clips_in_phase_batch.shape == (batch_size, M, F_in, layer_size)
            assert clips_out_of_phase_batch.shape == (batch_size, M, F_in, layer_size)

            clips_generated_in_phase_batch = g(
                clips=clips_in_phase_batch.reshape(batch_size * M, F_in, layer_size),
                styles=styles_repeated,
            )
            clips_generated_out_of_phase_batch = g(
                clips=clips_out_of_phase_batch.reshape(
                    batch_size * M, F_in, layer_size
                ),
                styles=styles_repeated,
            )

            assert clips_generated_in_phase_batch.shape == (
                batch_size * M,
                F_out,
                layer_size,
            )
            assert clips_generated_out_of_phase_batch.shape == (
                batch_size * M,
                F_out,
                layer_size,
            )
            clips_generated_in_phase_batch = clips_generated_in_phase_batch.reshape(
                batch_size,
                M,
                F_out,
                layer_size,
            )
            clips_generated_out_of_phase_batch = (
                clips_generated_out_of_phase_batch.reshape(
                    batch_size,
                    M,
                    F_out,
                    layer_size,
                )
            )

            clips_generated = overlap_add_combine(
                clips_generated_in_phase_batch,
                clips_generated_out_of_phase_batch,
                window_size=layer_size,
                window=window,
            )

            assert clips_generated.shape == (batch_size, F_out, N)

            clips_generated_batch = (
                clips_generated.reshape(batch_size, F_out, M, layer_size)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * M, F_out, layer_size)
            )
        else:  # if not overlap_and_add:
            assert clip.shape == (batch_size, F_in, N)
            clips_batch = (
                clip.reshape(batch_size, F_in, M, layer_size)
                .permute(0, 2, 1, 3)
                .reshape(batch_size * M, F_in, layer_size)
            )
            clips_generated_batch = g(clips_batch)
            assert clips_generated_batch.shape == (batch_size * M, F_out, layer_size)
            clips_generated = (
                clips_generated_batch.reshape(batch_size, M, F_out, layer_size)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, F_out, M * layer_size)
            )

        # clips_generated_in_phase_batch = window * clips_generated_in_phase_batch
        # clips_generated_out_of_phase_batch = window * clips_generated_out_of_phase_batch
        # # rotate clip to the right by half a layer
        # split_point_backward = N - (layer_size // 2)
        # clips_generated_in_phase = clips_generated_in_phase_batch.permute(
        #     1, 0, 2
        # ).reshape(F_out, N)
        # clips_generated_out_of_phase_rotated = (
        #     clips_generated_out_of_phase_batch.permute(1, 0, 2).reshape(F_out, N)
        # )
        # clips_generated_out_of_phase = torch.cat(
        #     (
        #         clips_generated_out_of_phase_rotated[:, split_point_backward:],
        #         clips_generated_out_of_phase_rotated[:, :split_point_backward],
        #     ),
        #     dim=1,
        # )
        # assert clips_generated_out_of_phase.shape == (F_out, N)
        # clips_generated_overlap_added = (
        #     clips_generated_in_phase + clips_generated_out_of_phase
        # )
        # clips_generated_batch = clips_generated_overlap_added.reshape(
        #     F_out, N // layer_size, layer_size
        # ).permute(1, 0, 2)

        assert clips_generated_batch.shape == (batch_size * M, F_out, layer_size)
        # generated.append(clips_generated_batch)
        if i + 1 < len(generators):
            current_factor = 1
            clip = clips_generated
            while current_factor < upsample_factor:
                clip = upsample_2x(clip, dim=2)
                current_factor *= 2
            assert current_factor == upsample_factor
            assert clip.shape == (batch_size, F_out, upsample_factor * N)

    return clip[:, :2]  # , generated


def sane_audio_loss(audio):
    assert isinstance(audio, torch.Tensor)
    mean = torch.mean(audio, dim=1, keepdim=True)
    zero_mean_audio = audio - mean
    mean_audio_amp = torch.mean(torch.abs(zero_mean_audio))
    return torch.mean(torch.abs(mean)) + torch.clamp(mean_audio_amp - 0.05, min=0.0)


def random_style_vector_batch(batch_size, num_style_features, device):
    return -1.0 + 2.0 * torch.rand(
        (
            batch_size,
            num_style_features,
        ),
        device=device,
    )


def random_style_vector(num_style_features, device):
    return random_style_vector_batch(
        batch_size=1, num_style_features=num_style_features, device=device
    ).squeeze(0)

def train(
    all_audio_clips,
    layer_size,
    upsample_factor,
    patch_size,
    generators,
    discriminators,
    # optimizer,
    # generator_optimizers,
    # discriminator_optimizers,
    generator_optimizer,
    discriminator_optimizer,
    num_levels,
    # num_features,
    num_input_style_features,
):
    assert isinstance(layer_size, int)
    assert isinstance(patch_size, int)
    assert isinstance(upsample_factor, int)
    # assert isinstance(generators, list)
    # assert isinstance(discriminators, list)
    # assert isinstance(generator_optimizers, list)
    # assert isinstance(discriminator_optimizers, list)
    assert isinstance(num_levels, int)
    # assert isinstance(num_features, int)
    assert isinstance(num_input_style_features, int)

    assert len(generators) == num_levels
    assert len(discriminators) == num_levels

    parallel_batch_size = 8  # 32  # 128
    sequential_batch_size = 1

    generator_losses = [0.0 for _ in range(num_levels)]
    discriminator_losses = [0.0 for _ in range(num_levels)]

    style_vectors = random_style_vector_batch(
        batch_size=parallel_batch_size,
        num_style_features=num_input_style_features,
        device="cuda",
    )

    sys.stdout.write("  ")
    sys.stdout.flush()

    n_critic = 3
    n_generator = 1
    n_all = n_critic + n_generator

    def layer_weight(i):
        return 1.0
        # return 1.0 / (1 + i)

    for i_batch in range(sequential_batch_size * n_all):
        mode = (i_batch // sequential_batch_size) % n_all
        first_step = (i_batch % sequential_batch_size) == 0
        last_step = ((i_batch + 1) % sequential_batch_size) == 0

        training_discriminator = mode < n_critic
        training_generator = not training_discriminator

        with torch.no_grad():
            training_clips_batch = random_training_batch(
                batch_size=parallel_batch_size,
                patch_size=patch_size,
                audio_clips=all_audio_clips,
            )
            real_sample_batches = downsample_many_times(
                clips_batch=training_clips_batch,
                layer_size=layer_size,
                downsample_factor=upsample_factor,
            )
        assert len(real_sample_batches) == num_levels
        assert all(
            [
                b.shape == (parallel_batch_size * upsample_factor ** i, 2, layer_size)
                for i, b in enumerate(real_sample_batches)
            ]
        )

        fake_clips = generate_full_audio_clip_batch(
            batch_size=parallel_batch_size,
            layer_size=layer_size,
            upsample_factor=upsample_factor,
            style_vectors=style_vectors,
            generators=generators,
            device="cuda",
            overlap_and_add=True,
        )
        assert fake_clips.shape == (parallel_batch_size, 2, patch_size)
        fake_sample_batches = downsample_many_times(
            clips_batch=fake_clips,
            layer_size=layer_size,
            downsample_factor=upsample_factor,
        )
        assert len(fake_sample_batches) == num_levels
        assert all(
            [
                b.shape == (parallel_batch_size * upsample_factor ** i, 2, layer_size)
                for i, (b, g) in enumerate(zip(fake_sample_batches, generators))
            ]
        )

        discriminator_real_predictions = [
            d(real) for d, real in zip(discriminators, real_sample_batches)
        ]

        discriminator_fake_predictions = [
            d(fake[:, :2]) for d, fake in zip(discriminators, fake_sample_batches)
        ]

        total_loss = 0.0

        if training_generator:
            assert len(discriminator_fake_predictions) == len(generator_losses)
            for i, pred_fake in enumerate(discriminator_fake_predictions):
                # l_i = layer_weight(i) * torch.mean(torch.square(pred_fake - 1.0))
                l_i = layer_weight(i) * -torch.mean(pred_fake)
                generator_losses[i] = l_i.detach().cpu().item()
                total_loss += l_i

            for fake_batch in fake_sample_batches:
                total_loss += sane_audio_loss(fake_batch)

            # clip = generate_full_audio_clip(
            #     layer_size=layer_size,
            #     num_features=num_features,
            #     style_vector=random_style_vector(
            #         num_style_features=num_input_style_features, device="cuda"
            #     ),
            #     generators=generators,
            #     device="cuda",
            # )
            # total_loss += torch.mean(
            #     torch.square(clip - all_audio_clips[0][0].to("cuda"))
            # )
            # generator_losses[0] = total_loss.item()

            model = generators
            optimizer = generator_optimizer

        if training_discriminator:
            assert len(discriminator_real_predictions) == len(discriminator_losses)
            for i, (pred_real, pred_fake) in enumerate(
                zip(discriminator_real_predictions, discriminator_fake_predictions)
            ):
                # l_i = layer_weight(i) * (
                #     torch.mean(torch.square(pred_real - 1.0) + torch.square(pred_fake))
                # )
                l_i = layer_weight(i) * (-torch.mean(pred_real) + torch.mean(pred_fake))
                discriminator_losses[i] = l_i.detach().cpu().item()
                # l_i += (torch.mean(torch.square(pred_real) + torch.square(pred_fake))) # regularizer
                total_loss += l_i
            model = discriminators
            optimizer = discriminator_optimizer

        if first_step:
            optimizer.zero_grad()

        total_loss.backward()

        if last_step:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad[...] /= sequential_batch_size
            optimizer.step()

        if training_discriminator and last_step:
            parameter_limit = 0.01
            with torch.no_grad():
                for p in discriminators.parameters():
                    p.clamp_(min=-parameter_limit, max=parameter_limit)

        console_animation_characters = "-\\|/"
        sys.stdout.write("\b")
        if i_batch + 1 < (sequential_batch_size * n_all):
            sys.stdout.write(
                console_animation_characters[
                    i_batch % len(console_animation_characters)
                ]
            )
        sys.stdout.flush()

    generator_losses = [l / sequential_batch_size for l in generator_losses]
    discriminator_losses = [l / sequential_batch_size for l in discriminator_losses]

    sys.stdout.write("\b\b")
    sys.stdout.flush()

    return (generator_losses, discriminator_losses)


def main():
    songs = []
    filenames = sorted(glob.glob("sound/*.flac"))
    # filenames = sorted(glob.glob("sound/adventure of a lifetime*.flac"))
    for i, filename in enumerate(filenames):
        progress_bar(i, len(filenames), "songs loaded")
        song_path, song_file = os.path.split(filename)
        song_name, song_ext = song_file.split(".")
        song_audio, sample_rate = torchaudio.load(filename)
        if song_audio.shape != (2, 65536):
            raise Exception(
                f'The file "{filename}" needs to be 2-channel stereo and 65536 sample long.'
            )
        songs.append((song_audio, song_name))
    # audio, sample_rate = torchaudio.load("sound/die taube auf dem dach 1.flac")
    # audio, sample_rate = torchaudio.load("sound/the shark 1.flac")
    audio_length = songs[0][0].shape[1]
    # print(f"audio.shape : {audio.shape}")
    # print(f"sample_rate : {sample_rate}")
    # plt.plot(audio[1])
    # plt.show()
    # sounddevice.play(audio.repeat(1, 10).permute(1, 0).numpy(), 32000)
    # time.sleep(10.0)

    # layer_size = 32
    # layer_size = 64
    # layer_size = 128
    # layer_size = 256
    layer_size = 512
    # layer_size = 1024
    # layer_size = 2048
    # layer_size = 8192
    # layer_size = 32768

    # patch_size = 8192
    # patch_size = 16384
    patch_size = 32768
    # patch_size = 65536

    upsample_factor = 4

    # num_features = 2
    # num_features = 4
    # num_features = 8
    # num_features = 16
    # num_features = 32
    # num_features = 64
    # num_features = 128

    # TODO: just use a basic fully-convolutional network architecture with dilated convolutions for upsampling
    # (keep existing code)

    num_input_style_features = 32
    num_hidden_style_features = 2

    generators_and_discriminators, num_levels = make_networks(
        layer_size=layer_size,
        patch_size=patch_size,
        upsample_factor=upsample_factor,
        max_hidden_features=32,
        min_hidden_features=2,
        # hidden_features=4,
        # hidden_features=8,
        # hidden_features=16,
        # hidden_features=32,
        # hidden_features=64,
        # kernel_size=5,
        # kernel_size=15,
        # kernel_size=31,
        # kernel_size=63,
        kernel_size=127,
        # kernel_size=255,
        # kernel_size=511,
        # positional_features=0,
        # positional_features=2,
        positional_features=0,
        generator_hidden_layers=1,
        discriminator_hidden_layers=1,
        num_input_style_features=num_input_style_features,
        num_hidden_style_features=num_hidden_style_features,
        device="cuda",
    )
    generators = generators_and_discriminators.generators
    discriminators = generators_and_discriminators.discriminators

    print(f"The network has {num_levels} levels")
    sys.stdout.write(f"{generators[0].num_input_features}")
    for g in generators:
        sys.stdout.write(f" -> {g.num_output_features}")
    sys.stdout.write("\n")
    sys.stdout.flush()

    lr = 0.0001

    # optimizer = torch.optim.Adam(generators_and_discriminators.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizer = torch.optim.SGD(generators_and_discriminators.parameters(), lr=lr)
    # optimizer = adabound.AdaBound(
    # generators_and_discriminators.parameters(), lr=lr,
    # final_lr=0.1
    # )

    # generator_optimizer = adabound.AdaBound(
    # generator_optimizer = torch.optim.SGD(
    # generator_optimizer = torch.optim.Adam(
    generator_optimizer = torch.optim.RMSprop(
        generators_and_discriminators.generators.parameters(),
        lr=lr,
        # weight_decay=1e-5,
        # betas=(0.8, 0.999)
        # final_lr=0.1
    )
    # discriminator_optimizer = adabound.AdaBound(
    # discriminator_optimizer = torch.optim.SGD(
    # discriminator_optimizer = torch.optim.Adam(
    discriminator_optimizer = torch.optim.RMSprop(
        generators_and_discriminators.discriminators.parameters(),
        lr=lr,
        # weight_decay=1e-5,
        # betas=(0.8, 0.999),
        # final_lr=0.1
    )

    # generator_optimizers = make_optimizers(generators, lr=lr)
    # discriminator_optimizers = make_optimizers(discriminators, lr=lr)

    # restore_module(generators_and_discriminators, "models/model_747.dat")
    # restore_module(generator_optimizer, "models/gen_opt_747.dat")
    # restore_module(discriminator_optimizer, "models/disc_opt_747.dat")
    generators_and_discriminators.train()

    plt.ion()

    fig, axes = plt.subplots(2, 3, figsize=(8, 8), dpi=80)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    ax_tl = axes[0, 0]
    ax_tm = axes[0, 1]
    ax_tr = axes[0, 2]
    ax_bl = axes[1, 0]
    ax_bm = axes[1, 1]
    ax_br = axes[1, 2]

    all_generator_losses = [[] for _ in range(num_levels)]
    all_discriminator_losses = [[] for _ in range(num_levels)]

    plot_interval = 64  # 256

    sounds_per_plot = 3

    def save_things(iteration):
        save_module(
            generators_and_discriminators,
            f"models/model_{iteration + 1}.dat",
        )
        save_module(
            generator_optimizer,
            f"models/gen_opt_{iteration + 1}.dat",
        )
        save_module(
            discriminator_optimizer,
            f"models/disc_opt_{iteration + 1}.dat",
        )

    try:
        for current_iteration in range(1_000_000):
            generator_losses, discriminator_losses = train(
                all_audio_clips=songs,
                layer_size=layer_size,
                patch_size=patch_size,
                upsample_factor=upsample_factor,
                generators=generators_and_discriminators.generators,
                discriminators=generators_and_discriminators.discriminators,
                # optimizer=optimizer,
                # generator_optimizers=generator_optimizers,
                # discriminator_optimizers=discriminator_optimizers,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                num_levels=num_levels,
                num_input_style_features=num_input_style_features,
            )

            for i_g, g_loss in enumerate(generator_losses):
                all_generator_losses[i_g].append(g_loss)

            for i_d, d_loss in enumerate(discriminator_losses):
                all_discriminator_losses[i_d].append(d_loss)

            time_to_plot = (
                (current_iteration + 1) % plot_interval
            ) == 0 or current_iteration <= 1

            if current_iteration != 0:
                progress_bar(
                    current_iteration % plot_interval,
                    plot_interval,
                    f"total iterations: {current_iteration + 1}",
                )

            if time_to_plot:
                with torch.no_grad():
                    output_clips = []
                    for _ in range(sounds_per_plot):
                        generated_clip = generate_full_audio_clip_batch(
                            batch_size=1,
                            device="cuda",
                            upsample_factor=upsample_factor,
                            style_vectors=random_style_vector_batch(
                                batch_size=1,
                                num_style_features=num_input_style_features,
                                device="cuda",
                            ),
                            generators=generators_and_discriminators.generators,
                            layer_size=layer_size,
                            overlap_and_add=True,
                        )
                        generated_clip = generated_clip.squeeze(0).detach().cpu()
                        output_clips.append(generated_clip)

                    audio_clip, song_name = random.choice(songs)

                    ax_tl.cla()
                    ax_tm.cla()
                    ax_tr.cla()
                    ax_bl.cla()
                    ax_bm.cla()
                    ax_br.cla()

                    ax_tl.scatter(
                        range(audio_clip.shape[1]), audio_clip[0].numpy(), s=1.0
                    )
                    ax_tl.scatter(
                        range(audio_clip.shape[1]), audio_clip[1].numpy(), s=1.0
                    )
                    ax_tl.set_ylim(-1.0, 1.0)

                    clip_to_plot = output_clips[0]

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

                    ax_tm.title.set_text(song_name)
                    ax_tm.imshow(
                        make_spectrogram(audio_clip[:2]).unsqueeze(2).repeat(1, 1, 3)
                    )

                    rgb_spectrogram = torch.stack(
                        [make_spectrogram(c) for c in output_clips[:3]], dim=2
                    )
                    ax_bm.imshow(rgb_spectrogram)

                    for i_g_loss, losses in enumerate(all_generator_losses):
                        t = i_g_loss / (num_levels - 1)
                        ax_tr.scatter(
                            range(len(losses)), losses, color=(0.0, t, 1.0 - t), s=1.0
                        )
                    ax_tr.set_xlim(-1, current_iteration + 1)
                    # ax_tr.set_yscale("log")

                    for i_d_loss, losses in enumerate(all_discriminator_losses):
                        t = i_d_loss / (num_levels - 1)
                        ax_br.scatter(
                            range(len(losses)), losses, color=(0.0, t, 1.0 - t), s=1.0
                        )
                    ax_br.set_xlim(-1, current_iteration + 1)
                    # ax_br.set_yscale("log")

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    plt_screenshot(fig).save(
                        f"images/image_{current_iteration + 1}.png"
                    )

                    for i_clip, clip in enumerate(output_clips):
                        clip_to_play = clip[:2]
                        clip_to_play = clip_to_play.detach()
                        mean_amp = torch.mean(clip_to_play, dim=1, keepdim=True)
                        clip_to_play -= mean_amp
                        max_amp = torch.max(
                            torch.abs(clip_to_play), dim=1, keepdim=True
                        )[0]
                        clip_to_play = 0.5 * clip_to_play / max_amp
                        wf.write(
                            f"output_sound/output_{current_iteration + 1}_v{i_clip + 1}.wav",
                            32000,
                            clip_to_play.repeat(1, 2).permute(1, 0).cpu().numpy(),
                        )

            if ((current_iteration + 1) % 4096) == 0:
                save_things(current_iteration)

                # return
    except KeyboardInterrupt as e:
        print("\n\nControl-C detected, saving model...\n")
        save_things(current_iteration)
        print("Exiting")
        exit(1)


if __name__ == "__main__":
    main()
