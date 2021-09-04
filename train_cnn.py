import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List, Tuple
import sys
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import random

from signal_processing import make_spectrogram, upsample_2x
from audio_util import load_audio_clips, random_audio_batch, sane_audio_loss
from torch_utils import (
    Apply,
    Downsample1d,
    FourierTransformLayer,
    InverseFourierTransformLayer,
    Log,
    ReshapeTensor,
    ResidualAdd,
    Upsample1d,
    WithNoise1d,
    make_conv_down,
    make_conv_same,
    make_conv_up,
    restore_module,
    save_module,
)
from util import assert_eq, flush, horizontal_rule, line, plt_screenshot
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
        colour_pale = (
            0.75 + 0.25 * self._colour_r,
            0.75 + 0.25 * self._colour_g,
            0.75 + 0.25 * self._colour_b,
        )
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


class TensorShape:
    def __init__(self, length, features: int):
        assert isinstance(length, int) or (length is None)
        assert isinstance(features, int)
        self.length = length
        self.features = features

    def as_tuple(self):
        if self.length is None:
            return (self.features,)
        else:
            return (self.features, self.length)

    def as_string(self):
        return " x ".join([str(s) for s in self.as_tuple()])


def assert_shape(x: torch.Tensor, s: TensorShape):
    expected = s.as_tuple()
    actual = x.shape[1:]
    if actual != expected:
        raise Exception(
            f"Expected tensor to have shape {expected}, but got {actual} instead"
        )


class Operation(metaclass=ABCMeta):
    @abstractmethod
    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        raise Exception("Not implemented")


class Convolution(Operation):
    def __init__(self, kernel_size: int):
        assert isinstance(kernel_size, int)
        self.kernel_size = kernel_size

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        assert input_shape.length is not None
        assert output_shape.length is not None
        f0 = input_shape.features
        n0 = input_shape.length
        f1 = output_shape.features
        n1 = output_shape.length
        if n0 < n1:
            assert (n1 % n0) == 0
            scale_factor = n1 // n0
            conv = make_conv_up(
                in_channels=f0,
                out_channels=f1,
                kernel_size=self.kernel_size,
                scale_factor=scale_factor,
            )
            desc = f"x {scale_factor} Dilated Convolution with kernel size {self.kernel_size}"
        elif n0 == n1:
            conv = make_conv_same(
                in_channels=f0,
                out_channels=f1,
                kernel_size=self.kernel_size,
                padding_mode="zeros",
            )
            desc = f"= Convolution with kernel size {self.kernel_size}"
        elif n0 > n1:
            assert (n0 % n1) == 0
            reduction_factor = n0 // n1
            conv = make_conv_down(
                in_channels=f0,
                out_channels=f1,
                kernel_size=self.kernel_size,
                reduction_factor=reduction_factor,
            )
            desc = (
                f"/ {reduction_factor} Convolution with kernel size {self.kernel_size}"
            )

        if n0 > 1:
            conv = nn.Sequential(nn.BatchNorm1d(num_features=f0), conv)
        return conv, desc


class ResidualConvolutionStack(Operation):
    def __init__(self, kernel_size: int):
        assert isinstance(kernel_size, int)
        self.kernel_size = kernel_size

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        n_in = input_shape.length
        f_in = input_shape.features
        n_out = output_shape.length
        f_out = output_shape.features
        assert n_in == n_out
        module = nn.Sequential(
            nn.BatchNorm1d(num_features=f_in),
            make_conv_same(
                in_channels=f_in,
                out_channels=f_out,
                kernel_size=self.kernel_size,
                padding_mode="circular",
            ),
            nn.BatchNorm1d(num_features=f_out),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # Apply(torch.sin),
            make_conv_same(
                in_channels=f_out,
                out_channels=f_out,
                kernel_size=self.kernel_size,
                padding_mode="circular",
            ),
            WithNoise1d(num_features=f_out),
        )
        return (
            ResidualAdd(module),
            f"Residual convolution with kernel size {self.kernel_size}",
        )


class LinearTransform(Operation):
    def __init__(self):
        pass

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        assert input_shape.length is None
        assert output_shape.length is None
        f0 = input_shape.features
        f1 = output_shape.features
        return nn.Linear(in_features=f0, out_features=f1), "Linear"


class Reshape(Operation):
    def __init__(self):
        pass

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        return ReshapeTensor(input_shape.as_tuple(), output_shape.as_tuple()), "Reshape"


class FourierTransform(Operation):
    def __init__(self):
        pass

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        assert input_shape.length is not None
        assert output_shape.length is not None
        f0 = input_shape.features
        n0 = input_shape.length
        f1 = output_shape.features
        n1 = output_shape.length
        if n0 == n1 * 2:
            # Forward
            assert f1 == f0 * 2
            return (
                FourierTransformLayer(num_features=f0, length=n0),
                "Fourier Transform",
            )
        elif n1 == n0 * 2:
            # Backward
            assert f0 == f1 * 2
            return (
                InverseFourierTransformLayer(num_features=f0, length=n0),
                "Inverse Fourier Transform",
            )
        else:
            raise Exception("Wat")


known_activation_functions = {
    "relu": nn.ReLU(),
    "leaky relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "sin": Apply(torch.sin),
}


class ActivationFunction(Operation):
    def __init__(self, name):
        assert name in known_activation_functions.keys()
        self.name = name

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        return known_activation_functions[self.name], self.name


class AddNoise(Operation):
    def __init__(self):
        pass

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        assert output_shape.length is not None
        return WithNoise1d(num_features=output_shape.features), "Add Noise"


class Resample(Operation):
    def __init__(self, mode):
        assert mode in ["linear", "nearest", "bicubic"]
        self.mode = mode
        pass

    def make_module(
        self, input_shape: TensorShape, output_shape: TensorShape
    ) -> Tuple[nn.Module, str]:
        assert input_shape.features == output_shape.features
        l_in = input_shape.length
        l_out = output_shape.length
        if l_in >= l_out:
            assert l_in % l_out == 0
            factor = l_in // l_out
            return Downsample1d(factor=factor, mode=self.mode), f"{factor}x Downsample"
        else:
            assert l_out % l_in == 0
            factor = l_out // l_in
            return Upsample1d(factor=factor, mode=self.mode), f"{factor}x Upsample"


class NeuralNetwork(nn.Module):
    def __init__(self, name: str, stages):
        super(NeuralNetwork, self).__init__()
        self.name = name
        self.sizes: List[TensorShape] = []

        self.operations: List[List[Operation]] = []

        assert isinstance(stages[0], TensorShape)
        self.sizes.append(stages[0])
        assert isinstance(stages[-1], TensorShape)
        current_operations = []
        for stage in stages[1:]:
            if isinstance(stage, Operation):
                current_operations.append(stage)
            elif isinstance(stage, TensorShape):
                assert len(current_operations) > 0
                self.operations.append(current_operations)
                current_operations = []
                self.sizes.append(stage)

        assert len(self.sizes) == (len(self.operations) + 1)

        modules: List[nn.Module] = []
        self.module_descriptions: List[List[str]] = []

        for i, ops in enumerate(self.operations):
            prev_size = self.sizes[i]
            next_size = self.sizes[i + 1]
            current_modules: List[nn.Module] = []
            current_descriptions: List[str] = []
            for op in ops:
                module, description = op.make_module(prev_size, next_size)
                current_modules.append(module)
                current_descriptions.append(description)
            modules.append(nn.Sequential(*current_modules))
            self.module_descriptions.append(current_descriptions)
        self.module_list = nn.Sequential(*modules)

    def print_description(self):
        horizontal_rule()
        line()
        line(self.name)
        line()
        initial_size = self.sizes[0]
        line(f"[ {initial_size.as_string()} ]")
        for i in range(len(self.module_list)):
            prev_size = self.sizes[i]
            next_size = self.sizes[i + 1]
            for d in self.module_descriptions[i]:
                line(d)
            line(f"[ {next_size.as_string()} ]")
        line()
        horizontal_rule()
        print()

    def forward(self, x):
        assert isinstance(x, torch.Tensor)

        assert_shape(x, self.sizes[0])
        # print("------------------")
        for i, mods in enumerate(self.module_list):
            prev_size = self.sizes[i]
            next_size = self.sizes[i + 1]
            for j, m in enumerate(mods):
                # print(f"{x.shape} -> {self.module_descriptions[i][j]}")
                x = m(x)
                # print(f" -> {x.shape}")
            assert_shape(x, next_size)

        return x


def random_initial_vector(batch_size, num_features):
    return -1.0 + 2.0 * torch.randn(
        (batch_size, num_features), dtype=torch.float32, device="cuda"
    )
    # return -1.0 + 2.0 * torch.rand(
    #     (batch_size, num_features, num_samples), dtype=torch.float32, device="cuda"
    # )


# class SimpleConvolutionalGenerator(nn.Module):
#     def __init__(self, architecture):
#         super(SimpleConvolutionalGenerator, self).__init__()
#         assert isinstance(architecture, Architecture)

#         modules = []

#         self.architecture = architecture

#         horizontal_rule()
#         line("Generator Architecture")
#         line()

#         initial_shape = architecture.sizes[0]
#         final_shape = architecture.sizes[-1]

#         line(f"[ {initial_shape.features} x {initial_shape.length} ]")

#         for i in range(len(architecture.operations)):
#             f_prev, n_prev = architecture.sizes[i]
#             f_next, n_next = architecture.sizes[i + 1]
#             op = architecture.operations[i]
#             # modules.append(Log(f"Layer {i} begin, expecting [{f_prev}, {n_prev}]"))
#             assert_eq((n_next % n_prev), 0)
#             if n_prev > 1:
#                 line("Batch Normalization")
#                 modules.append(nn.BatchNorm1d(num_features=f_prev))
#             if isinstance(op, \Convolution):
#                 line(f"x {n_next // n_prev} Dilated Convolution")
#                 make_conv_up(
#                     in_channels=f_prev,
#                     out_channels=f_next,
#                     kernel_size=k_prev,
#                     # padding_mode="circular",
#                     scale_factor=(n_next // n_prev),
#                 )
#             line("Noise")
#             modules.append(WithNoise1d(num_features=f_next))
#             if i + 1 < len(previous_and_next_params):
#                 modules.append(hidden_activation_function_generator)
#                 line("Activation Function")
#             # modules.append(Log(f"Layer {i} end, expecting [{f_next}, {n_next}]"))
#             line(f"[ {f_next} x {n_next} ]")

#         line()
#         horizontal_rule()
#         flush()

#         self.model = nn.Sequential(*modules)

#     def forward(self, x):
#         assert isinstance(x, torch.Tensor)
#         B = x.shape[0]
#         assert_eq(x.shape[1:], (self.initial_features, self.initial_samples))

#         y = self.model(x)

#         assert_eq(y.shape, (B, self.final_features, self.final_samples))

#         return y


# class SimpleConvolutionalDiscriminator(nn.Module):
#     def __init__(self, features_lengths_kernel_sizes):
#         super(SimpleConvolutionalDiscriminator, self).__init__()

#         assert isinstance(features_lengths_kernel_sizes, list)
#         assert all(
#             [
#                 (isinstance(f, int) and isinstance(n, int) and isinstance(k, int))
#                 for (f, n, k) in features_lengths_kernel_sizes
#             ]
#         )

#         modules = []

#         f_0, n_0, _ = features_lengths_kernel_sizes[0]
#         f_n, n_n, _ = features_lengths_kernel_sizes[-1]

#         self.initial_features = f_0
#         self.initial_samples = n_0
#         self.final_features = f_n
#         self.final_samples = n_n

#         previous_and_next_params = zip(
#             features_lengths_kernel_sizes, features_lengths_kernel_sizes[1:]
#         )

#         horizontal_rule()
#         line("Discriminator Architecture")
#         line()

#         line(f"[ {f_0} x {n_0} ]")

#         for i, ((fs_prev, fs_next)) in enumerate(previous_and_next_params):
#             f_prev, n_prev, _ = fs_prev
#             f_next, n_next, k_next = fs_next
#             assert_eq((n_prev % n_next), 0)
#             # modules.append(Log(f"Layer {i} begin, expecting [{f_prev}, {n_prev}]"))
#             if n_prev > 1:
#                 line("Batch Normalization")
#                 modules.append(nn.BatchNorm1d(num_features=f_prev))
#             line(f"/ {n_prev // n_next} Convolution")
#             line("Activation Function")
#             modules.extend(
#                 [
#                     # Downsample1d(factor=(n_prev // n_next)),
#                     # make_conv_same(
#                     make_conv_down(
#                         in_channels=f_prev,
#                         out_channels=f_next,
#                         kernel_size=k_next,
#                         # padding_mode="circular",
#                         reduction_factor=(n_prev // n_next),
#                     ),
#                     hidden_activation_function_discriminator,
#                 ]
#             )
#             # modules.append(Log(f"Layer {i} end, expecting [{f_next}, {n_next}]"))
#             line(f"[ {f_next} x {n_next} ]")

#         self.model = nn.Sequential(*modules)
#         self.final = nn.Linear(in_features=(f_n * n_n), out_features=1)
#         line("Reshape")
#         line(f"[ {f_n * n_n} ]")
#         line("Linear")
#         line(f"[ {1} ]")

#         line()
#         horizontal_rule()
#         flush()

#     def forward(self, x):
#         assert isinstance(x, torch.Tensor)
#         B = x.shape[0]
#         assert_eq(x.shape[1:], (self.initial_features, self.initial_samples))

#         y = self.model(x)

#         assert_eq(y.shape, (B, self.final_features, self.final_samples))

#         y_flat = y.reshape(B, self.final_features * self.final_samples)

#         z = self.final(y_flat)

#         assert_eq(z.shape, (B, 1))

#         return z


# class GeneratorAndDiscriminator(nn.Module):
#     def __init__(self, features_lengths_kernel_sizes):
#         super(GeneratorAndDiscriminator, self).__init__()
#         assert isinstance(features_lengths_kernel_sizes, list)
#         self.generator = SimpleConvolutionalGenerator(features_lengths_kernel_sizes)
#         features_lengths_kernel_sizes_reversed = list(
#             features_lengths_kernel_sizes[::-1]
#         )
#         self.discriminator = SimpleConvolutionalDiscriminator(
#             features_lengths_kernel_sizes_reversed
#         )


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
    songs = load_audio_clips("sound/*.flac")

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

    # TODO:
    # - read about torch.stft and torch.istft
    #   https://pytorch.org/docs/stable/generated/torch.stft.html#torch.stft
    #   https://pytorch.org/docs/stable/generated/torch.istft.html#torch.istft

    # generator_architecture = [
    #     TensorShape(length=None, features=latent_features),
    #     LinearTransform(),
    #     ActivationFunction("sin"),
    #     TensorShape(length=None, features=64),
    #     LinearTransform(),
    #     ActivationFunction("sin"),
    #     TensorShape(length=None, features=64),
    #     LinearTransform(),
    #     # ActivationFunction(activation_function_generator),
    #     TensorShape(length=None, features=(512 * 8)),
    #     Reshape(),
    #     TensorShape(length=512, features=8),
    #     FourierTransform(),
    #     TensorShape(length=1024, features=4),
    #     Convolution(kernel_size=127),
    #     AddNoise(),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=4096, features=64),
    #     Convolution(kernel_size=15),
    #     AddNoise(),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=16384, features=16),
    #     Convolution(kernel_size=15),
    #     AddNoise(),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=65536, features=4),
    #     Convolution(kernel_size=15),
    #     AddNoise(),
    #     TensorShape(length=65536, features=2),
    # ]

    # discriminator_architecture = [
    #     TensorShape(length=65536, features=2),
    #     Convolution(kernel_size=15),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=65536, features=4),
    #     Convolution(kernel_size=15),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=16384, features=16),
    #     Convolution(kernel_size=15),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=4096, features=64),
    #     Convolution(kernel_size=127),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=1024, features=4),
    #     FourierTransform(),
    #     TensorShape(length=512, features=8),
    #     Reshape(),
    #     TensorShape(length=None, features=(512 * 8)),
    #     LinearTransform(),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=None, features=(64)),
    #     LinearTransform(),
    #     ActivationFunction("leaky relu"),
    #     TensorShape(length=None, features=(64)),
    #     LinearTransform(),
    #     TensorShape(length=None, features=1),
    # ]

    generator_architecture = [
        TensorShape(length=None, features=latent_features),
        LinearTransform(),
        ActivationFunction("leaky relu"),
        TensorShape(length=None, features=256),
        LinearTransform(),
        TensorShape(length=None, features=(1024 * 8)),
        Reshape(),
        TensorShape(length=1024, features=8),
        FourierTransform(),
        TensorShape(length=2048, features=4),
        ResidualConvolutionStack(kernel_size=31),
        TensorShape(length=2048, features=16),
        Resample(mode="linear"),
        TensorShape(length=4096, features=16),
        ResidualConvolutionStack(kernel_size=31),
        TensorShape(length=4096, features=16),
        Resample(mode="linear"),
        TensorShape(length=16384, features=16),
        ResidualConvolutionStack(kernel_size=63),
        TensorShape(length=16384, features=4),
        Resample(mode="linear"),
        TensorShape(length=65536, features=4),
        ResidualConvolutionStack(kernel_size=127),
        TensorShape(length=65536, features=2),
    ]
 
    discriminator_architecture = [
        TensorShape(length=65536, features=2),
        Convolution(kernel_size=127),
        ActivationFunction("leaky relu"),
        TensorShape(length=16384, features=16),
        Convolution(kernel_size=63),
        ActivationFunction("leaky relu"),
        TensorShape(length=4096, features=16),
        Convolution(kernel_size=31),
        ActivationFunction("leaky relu"),
        TensorShape(length=2048, features=4),
        FourierTransform(),
        TensorShape(length=1024, features=8),
        Reshape(),
        TensorShape(length=None, features=(1024 * 8)),
        LinearTransform(),
        ActivationFunction("leaky relu"),
        TensorShape(length=None, features=(256)),
        LinearTransform(),
        TensorShape(length=None, features=1),
    ]

    generator = NeuralNetwork("Generator", generator_architecture).cuda()
    discriminator = NeuralNetwork("Discriminator", discriminator_architecture).cuda()

    generator.print_description()
    discriminator.print_description()

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
