from torch_utils import slice_along_dim
import scipy.signal
import torch
import numpy as np


def make_spectrogram(x, normalize=True):
    assert len(x.shape) == 2
    assert x.shape[1] > x.shape[0]
    x = x.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    sgs = []
    for i in range(x.shape[0]):
        xi = x[i]
        _, _, sg = scipy.signal.spectrogram(
            xi.detach().cpu().numpy(),
            window="blackmanharris",
            nperseg=256,
            noverlap=128,
        )
        sg = np.nan_to_num(sg, nan=0.0, posinf=0.0, neginf=0.0)
        sg = np.log(np.clip(np.abs(sg), a_min=1e-6, a_max=128))
        assert len(sg.shape) == 2
        sgs.append(torch.tensor(sg))
    spectrogram = torch.cat(sgs, dim=0).flip(0)
    if normalize:
        spectrogram_min = torch.min(spectrogram)
        spectrogram_max = torch.max(spectrogram)
        if spectrogram_max > spectrogram_min:
            spectrogram = (spectrogram - spectrogram_min) / (
                spectrogram_max - spectrogram_min
            )
        else:
            spectrogram.fill_(0.0)
    return spectrogram


def cosine_window(size, device):
    assert isinstance(size, int)
    ls = torch.tensor(
        np.linspace(0.0, 1.0, num=size, endpoint=False),
        dtype=torch.float,
        device=device,
    )
    assert ls.shape == (size,)
    return 0.5 - 0.5 * torch.cos(ls * 2.0 * np.pi)


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


def downsample_2x(x, dim):
    assert isinstance(x, torch.Tensor)
    assert isinstance(dim, int)
    num_dims = len(x.shape)
    assert dim < num_dims
    evens = x[slice_along_dim(num_dims, dim, start=0, step=2)]
    odds = x[slice_along_dim(num_dims, dim, start=1, step=2)]
    return 0.5 * (evens + odds)
