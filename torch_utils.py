import os
from torch import fft
import torchaudio
import torchaudio.transforms
from util import assert_eq
import torch
import torch.nn as nn
import math
from functools import reduce


def slice_along_dim(num_dims, dim, start=None, stop=None, step=None):
    assert isinstance(num_dims, int)
    begin = (slice(None),) * dim
    middle = (slice(start, stop, step),)
    end = (slice(None),) * (num_dims - dim - 1)
    ret = begin + middle + end
    assert_eq(len(ret), num_dims)
    return ret


def save_module(the_module, filename):
    print(f'Saving module to "{filename}"')
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(the_module.state_dict(), filename)


def restore_module(the_module, filename):
    print('Restoring module from "{}"'.format(filename))
    the_module.load_state_dict(torch.load(filename))


def make_conv_same(in_channels, out_channels, kernel_size, padding_mode):
    assert isinstance(in_channels, int)
    assert isinstance(out_channels, int)
    assert isinstance(kernel_size, int)
    assert isinstance(padding_mode, str)
    padding = (kernel_size - 1) // 2
    assert padding >= 0
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        padding_mode=padding_mode,
    )


def make_conv_up(in_channels, out_channels, kernel_size, scale_factor):
    assert scale_factor <= kernel_size
    # assert ((kernel_size - scale_factor) % 2) == 0
    padding = (kernel_size - scale_factor + 1) // 2
    output_padding = (kernel_size - scale_factor) % 2
    assert padding >= 0
    assert output_padding >= 0
    return nn.ConvTranspose1d(
        in_channels=in_channels,  # good
        out_channels=out_channels,  # good
        kernel_size=kernel_size,  # good
        stride=scale_factor,  # good
        padding=padding,
        output_padding=output_padding,
    )


def make_conv_down(in_channels, out_channels, kernel_size, reduction_factor):
    assert kernel_size % 2 == 1
    padding = (kernel_size - reduction_factor + 1) // 2
    assert padding >= 0
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=reduction_factor,
        padding=padding,
    )


def make_positional_encoding(features, length, frequency_multiplier=1, device="cpu"):
    assert isinstance(features, int)
    assert isinstance(length, int)
    assert isinstance(frequency_multiplier, int)
    t = torch.linspace(
        start=0.0,
        end=((length - 1) / length),
        steps=length,
        dtype=torch.float,
        device=device,
    )
    f = torch.linspace(
        start=1.0,
        end=(features * frequency_multiplier),
        steps=(features // 2),
        dtype=torch.float,
        device=device,
    )

    phases = math.tau * t[None, :] * f[:, None]
    assert phases.shape == (features // 2, length)

    pos_enc = torch.cat([torch.cos(phases), torch.sin(phases)], dim=0)

    assert pos_enc.shape == (features, length)

    pos_enc.requires_grad_(False)

    return pos_enc


def prod(iterable, start=1):
    return reduce(lambda a, b: a * b, iterable, start)


class Reshape(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Reshape, self).__init__()
        assert isinstance(input_shape, tuple) or isinstance(input_shape, torch.Size)
        assert isinstance(output_shape, tuple) or isinstance(output_shape, torch.Size)
        assert prod(input_shape) == prod(output_shape)
        self._input_shape = input_shape
        self._output_shape = output_shape

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        if x.shape[1:] != self._input_shape:
            raise Exception(
                f"Reshape: expected input size {self._input_shape} but got {tuple(x.shape[1:])} instead"
            )
        return x.view(B, *self._output_shape)


class CheckShape(nn.Module):
    def __init__(self, expected_shape):
        super(CheckShape, self).__init__()
        assert isinstance(expected_shape, tuple) or isinstance(
            expected_shape, torch.Size
        )
        self._expected_shape = expected_shape

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        if x.shape[1:] != self._expected_shape:
            raise Exception(
                f"CheckShape: expected input size {self._expected_shape} but got {tuple(x.shape[1:])} instead"
            )
        return x


class WithNoise1d(nn.Module):
    def __init__(self, num_features):
        super(WithNoise1d, self).__init__()
        assert isinstance(num_features, int)
        self.num_features = num_features
        self.weights = nn.parameter.Parameter(
            0.01 * torch.ones((num_features,), dtype=torch.float)
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B, F, N = x.shape
        assert_eq(F, self.num_features)
        noise = torch.randn((B, F, N), dtype=x.dtype, device=x.device)
        weight = self.weights.reshape(1, F, 1)
        return x + weight * noise


class WithNoise2d(nn.Module):
    def __init__(self, num_features):
        super(WithNoise2d, self).__init__()
        assert isinstance(num_features, int)
        self.num_features = num_features
        self.weights = nn.parameter.Parameter(
            0.01 * torch.ones((num_features,), dtype=torch.float)
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B, F, H, W = x.shape
        assert_eq(F, self.num_features)
        noise = torch.randn((B, F, H, W), dtype=x.dtype, device=x.device)
        weight = self.weights.reshape(1, F, 1, 1)
        return x + weight * noise


class ConcatenateConstant(nn.Module):
    def __init__(self, constant):
        super(ConcatenateConstant, self).__init__()
        assert isinstance(constant, torch.Tensor)
        assert_eq(len(constant.shape), 2)
        self.constant = constant

    def forward(self, x):
        assert_eq(x.shape[2:], self.constant.shape[1:])
        B = x.shape[0]
        return torch.cat((x, self.constant.unsqueeze(0).repeat(B, 1, 1)), dim=1)


class Apply(nn.Module):
    def __init__(self, fn, *args, **kwargs):
        super(Apply, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.fn(x, *self.args, **self.kwargs)


class FourierTransformLayer(nn.Module):
    def __init__(self, num_features, length):
        super(FourierTransformLayer, self).__init__()
        assert isinstance(num_features, int)
        assert isinstance(length, int)
        self.num_features = num_features
        self.length = length

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        assert_eq(x.shape, (B, self.num_features, self.length))
        x_fd = fft.rfft(x, n=self.length, dim=2, norm="ortho")
        assert_eq(x_fd.shape, (B, self.num_features, (self.length // 2 + 1)))
        x_fd_but_without_the_nyquist = x_fd[:, :, : (self.length // 2)]
        x_fd_real = torch.real(x_fd_but_without_the_nyquist)
        x_fd_imag = torch.imag(x_fd_but_without_the_nyquist)
        assert_eq(x_fd_real.shape, (B, self.num_features, (self.length // 2)))
        assert_eq(x_fd_imag.shape, (B, self.num_features, (self.length // 2)))
        y = torch.cat((x_fd_real, x_fd_imag), dim=1)
        assert_eq(y.shape, (B, (self.num_features * 2), (self.length // 2)))
        return y


class InverseFourierTransformLayer(nn.Module):
    def __init__(self, num_features, length):
        super(InverseFourierTransformLayer, self).__init__()
        assert isinstance(num_features, int)
        assert isinstance(length, int)
        self.num_features = num_features
        self.length = length

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B = x.shape[0]
        assert_eq(x.shape, (B, self.num_features, self.length))
        x_with_nyquist = torch.cat((x, torch.zeros_like(x[:, :, :1])), dim=2)
        assert_eq(x_with_nyquist.shape, (B, self.num_features, self.length + 1))
        x_real = x_with_nyquist[:, : (self.num_features // 2)]
        x_imag = x_with_nyquist[:, (self.num_features // 2) :]
        x_complex = torch.complex(real=x_real, imag=x_imag)
        assert_eq(x_complex.shape, (B, self.num_features // 2, self.length + 1))
        x_td = fft.irfft(x_complex, n=(self.length * 2), dim=2, norm="ortho")
        assert_eq(x_td.shape, (B, (self.num_features // 2), (self.length * 2)))
        return x_td


s_log_layers_enabled = False


def enable_log_layers():
    global s_log_layers_enabled
    s_log_layers_enabled = True


def disable_log_layers():
    global s_log_layers_enabled
    s_log_layers_enabled = False


class Log(nn.Module):
    def __init__(self, description=""):
        super(Log, self).__init__()
        assert isinstance(description, str)
        self._description = description

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        if s_log_layers_enabled:
            print(f"Log Layer - {self._description}")
            print(f"    Batch size:   {x.shape[0]}")
            print(f"    Tensor shape: {x.shape[1:]}")
            print("")
        return x


class Resample1d(nn.Module):
    def __init__(self, new_length, mode="linear"):
        super(Resample1d, self).__init__()
        assert isinstance(new_length, int)
        assert mode in ["linear", "nearest", "bicubic"]
        self.new_length = new_length
        self.mode = mode

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B, F, N = x.shape
        return nn.functional.interpolate(
            x,
            size=self.new_length,
            mode=self.mode,
            # Dammit, pytorch
            **({} if self.mode == "nearest" else {"align_corners": False}),
        )


class ResidualAdd1d(nn.Module):
    def __init__(self, model):
        super(ResidualAdd1d, self).__init__()
        assert isinstance(model, nn.Module)
        self.model = model

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        B1, F1, N1 = x.shape
        y = self.model(x)
        B2, F2, N2 = y.shape
        assert_eq(B1, B2)
        assert_eq(N1, N2)
        if F1 > F2:
            return x[:, :F2] + y
        else:
            return torch.cat([x + y[:, :F1], y[:, F1:]], dim=1)


class CircularPad1d(nn.Module):
    def __init__(self, amount):
        super(CircularPad1d, self).__init__()
        assert isinstance(amount, int)
        self.amount = amount

    def forward(self, x):
        batch_size, num_features, length = x.shape
        assert length > self.amount
        padded = torch.cat([x[:, :, -self.amount :], x, x[:, :, : self.amount],], dim=2)
        assert padded.shape == (batch_size, num_features, length + (2 * self.amount))
        return padded


class CircularDownSampleAA(nn.Module):
    def __init__(self, factor):
        super(CircularDownSampleAA, self).__init__()
        assert isinstance(factor, int)
        self.pad_ratio = 16
        self.factor = factor
        self.pad = CircularPad1d(factor * self.pad_ratio)
        self.transform = torchaudio.transforms.Resample(orig_freq=factor, new_freq=1)

    def forward(self, x):
        B, C, L = x.shape
        padded = self.pad(x)
        assert padded.shape == (B, C, L + 2 * self.pad_ratio * self.factor)
        resampled = self.transform(padded)
        assert resampled.shape == (B, C, L // self.factor + 2 * self.pad_ratio)
        final = resampled[:, :, self.pad_ratio : -self.pad_ratio]
        assert final.shape == (B, C, L // self.factor)
        return final


class CircularUpSampleAA(nn.Module):
    def __init__(self, factor):
        super(CircularUpSampleAA, self).__init__()
        assert isinstance(factor, int)
        self.pad_ratio = 16
        self.factor = factor
        self.pad = CircularPad1d(self.pad_ratio)
        self.transform = torchaudio.transforms.Resample(orig_freq=1, new_freq=factor)

    def forward(self, x):
        B, C, L = x.shape
        padded = self.pad(x)
        assert padded.shape == (B, C, L + 2 * self.pad_ratio)
        resampled = self.transform(padded)
        assert resampled.shape == (B, C, (L + 2 * self.pad_ratio) * self.factor)
        final = resampled[
            :, :, (self.factor * self.pad_ratio) : -(self.factor * self.pad_ratio)
        ]
        assert final.shape == (B, C, L * self.factor)
        return final
