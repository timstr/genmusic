import math
import torch
import torch.nn as nn

from torch_utils import CircularDownSampleAA, CircularUpSampleAA, CircularPad1d, make_positional_encoding


def _make_convolutional_layers(
    input_features, hidden_features, output_features, num_layers=1
):
    kernel_size = 15
    padding = (kernel_size - 1) // 2
    modules = [
        CircularPad1d(amount=padding),
        nn.Conv1d(
            in_channels=input_features,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            # padding=padding,
            # padding_mode="zeros",
            stride=1,
        ),
        nn.BatchNorm1d(num_features=hidden_features),
        nn.GELU(),
    ]
    for _ in range(num_layers - 1):
        modules.extend(
            [
                CircularPad1d(amount=padding),
                nn.Conv1d(
                    in_channels=hidden_features,
                    out_channels=hidden_features,
                    kernel_size=kernel_size,
                    # padding=padding,
                    # padding_mode="zeros",
                    stride=1,
                ),
                nn.BatchNorm1d(num_features=hidden_features),
                nn.GELU(),
            ]
        )
    modules.extend(
        [
            CircularPad1d(amount=padding),
            nn.Conv1d(
                in_channels=hidden_features,
                out_channels=output_features,
                kernel_size=kernel_size,
                # padding=padding,
                # padding_mode="zeros",
                stride=1,
            ),
        ]
    )
    return nn.Sequential(*modules)


def _add_positional_encoding(x):
    batch_size, features, length = x.shape
    pos_enc = make_positional_encoding(
        features=features,
        length=length,
        frequency_multiplier=1,
        device=x.device
    )
    return x + pos_enc[None]

g_plot_count = 0

class AudioDiffusionNetV2(nn.Module):
    def __init__(self):
        super(AudioDiffusionNetV2, self).__init__()

        self.initial_cnn = _make_convolutional_layers(
            input_features=2, hidden_features=8, output_features=8,
        )

        # 65536
        self.downsample_1 = CircularDownSampleAA(2)
        self.down_cnn_1 = _make_convolutional_layers(
            input_features=8, hidden_features=8, output_features=8
        )
        # 32768
        self.downsample_2 = CircularDownSampleAA(2)
        self.down_cnn_2 = _make_convolutional_layers(
            input_features=8, hidden_features=8, output_features=8
        )
        # 16384
        self.downsample_3 = CircularDownSampleAA(2)
        self.down_cnn_3 = _make_convolutional_layers(
            input_features=8, hidden_features=16, output_features=16
        )
        # 8192
        self.downsample_4 = CircularDownSampleAA(2)
        self.down_cnn_4 = _make_convolutional_layers(
            input_features=16, hidden_features=32, output_features=32
        )
        # 4096
        self.downsample_5 = CircularDownSampleAA(2)
        self.down_cnn_5 = _make_convolutional_layers(
            input_features=32, hidden_features=64, output_features=64
        )
        # 2048
        self.downsample_6 = CircularDownSampleAA(2)
        self.down_cnn_6 = _make_convolutional_layers(
            input_features=64, hidden_features=128, output_features=128
        )
        # 1024
        self.downsample_7 = CircularDownSampleAA(2)
        self.down_cnn_7 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=128
        )
        # 512
        self.downsample_8 = CircularDownSampleAA(2)
        self.down_cnn_8 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=128
        )
        # 256
        self.downsample_9 = CircularDownSampleAA(2)
        self.down_cnn_9 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=128
        )
        # 128 ----------------------------------------------
        self.upsample_9 = CircularUpSampleAA(2)
        self.up_cnn_9 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=128
        )
        pe_9 = make_positional_encoding(features=128, length=128)
        # 256
        self.upsample_8 = CircularUpSampleAA(2)
        pe_8 = make_positional_encoding(features=128, length=256)
        self.up_cnn_8 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=128
        )
        # 512
        self.upsample_7 = CircularUpSampleAA(2)
        pe_7 = make_positional_encoding(features=128, length=512)
        self.up_cnn_7 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=128
        )
        # 1024
        self.upsample_6 = CircularUpSampleAA(2)
        pe_6 = make_positional_encoding(features=128, length=1024)
        self.up_cnn_6 = _make_convolutional_layers(
            input_features=128, hidden_features=128, output_features=64
        )
        # 2048
        self.upsample_5 = CircularUpSampleAA(2)
        self.up_cnn_5 = _make_convolutional_layers(
            input_features=64, hidden_features=32, output_features=32
        )
        # 4096
        self.upsample_4 = CircularUpSampleAA(2)
        self.up_cnn_4 = _make_convolutional_layers(
            input_features=32, hidden_features=16, output_features=16
        )
        # 8192
        self.upsample_3 = CircularUpSampleAA(2)
        self.up_cnn_3 = _make_convolutional_layers(
            input_features=16, hidden_features=8, output_features=8
        )
        # 16384
        self.upsample_2 = CircularUpSampleAA(2)
        self.up_cnn_2 = _make_convolutional_layers(
            input_features=8, hidden_features=8, output_features=8
        )
        # 32768
        self.upsample_1 = CircularUpSampleAA(2)
        self.up_cnn_1 = _make_convolutional_layers(
            input_features=8, hidden_features=8, output_features=8
        )
        # 65536

        self.final_cnn = _make_convolutional_layers(
            input_features=8, hidden_features=8, output_features=2
        )

        self.pe_9 = nn.parameter.Parameter(data=pe_9, requires_grad=False)
        self.pe_8 = nn.parameter.Parameter(data=pe_8, requires_grad=False)
        self.pe_7 = nn.parameter.Parameter(data=pe_7, requires_grad=False)
        self.pe_6 = nn.parameter.Parameter(data=pe_6, requires_grad=False)

    def forward(self, x, t, plot_activations=False):
        assert isinstance(x, torch.Tensor)
        batch_size, num_channels, length = x.shape
        assert num_channels == 2
        assert length == 65536
        assert t.shape == (batch_size,)

        tensors_to_plot = []
        def f(tensor):
            if plot_activations:
                tensors_to_plot.append(tensor[0].detach().cpu())
            return tensor

        def plot_tensors():
            global g_plot_count
            if not plot_activations:
                return
            # print("Plotting intermediate tensor values...")
            import matplotlib.pyplot as plt
            import os
            num_tensors = len(tensors_to_plot)
            fig = plt.figure(2, figsize=(16.0, num_tensors))
            ax = fig.subplots(num_tensors, 1)
            for i, tensor in enumerate(tensors_to_plot):
                # ax[i].set_box_aspect(1.0 / 16.0)
                ax[i].axis("off")
                ax[i].imshow(
                    tensor.reshape(-1, tensor.shape[-1]),
                    vmin=-2.0,
                    vmax=2.0,
                    aspect="auto",
                    # cmap="seismic",
                    cmap="twilight",
                )
            fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.99))
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.savefig(os.path.join("images", f"activations_{g_plot_count}.png"))
            g_plot_count += 1
            # print("Plotting intermediate tensor values... Done")

        x0 = f(self.initial_cnn(f(x)))

        x1 = f(self.down_cnn_1(self.downsample_1(x0)))
        x2 = f(self.down_cnn_2(self.downsample_2(x1)))
        x3 = f(self.down_cnn_3(self.downsample_3(x2)))
        x4 = f(self.down_cnn_4(self.downsample_4(x3)))
        x5 = f(self.down_cnn_5(self.downsample_5(x4)))
        x6 = f(self.down_cnn_6(self.downsample_6(x5)))
        x7 = f(self.down_cnn_7(self.downsample_7(x6)))
        x8 = f(self.down_cnn_8(self.downsample_8(x7)))
        z = f(self.down_cnn_9(self.downsample_9(x8)))

        hidden_features = z.shape[1]
        hidden_length = z.shape[2]

        t_frequencies = torch.linspace(
            start=1.0,
            end=hidden_features,
            steps=(hidden_features // 2),
            device=z.device,
        )

        t_phases = math.tau * t[:, None] * t_frequencies[None, :]
        assert t_phases.shape == (batch_size, (hidden_features // 2))

        t_positional_encoding = torch.cat(
            [torch.cos(t_phases), torch.sin(t_phases)], dim=1
        )
        assert t_positional_encoding.shape == (batch_size, hidden_features)
        t_positional_encoding.requires_grad_(False)

        z_and_t = z + t_positional_encoding[:, :, None]

        y8 = f(x8 + self.upsample_9(self.up_cnn_9(z_and_t + self.pe_9)))
        y7 = f(x7 + self.upsample_8(self.up_cnn_8(y8 + self.pe_8)))
        y6 = f(x6 + self.upsample_7(self.up_cnn_7(y7 + self.pe_7)))
        y5 = f(x5 + self.upsample_6(self.up_cnn_6(y6 + self.pe_6)))
        y4 = f(x4 + self.upsample_5(self.up_cnn_5(y5)))
        y3 = f(x3 + self.upsample_4(self.up_cnn_4(y4)))
        y2 = f(x2 + self.upsample_3(self.up_cnn_3(y3)))
        y1 = f(x1 + self.upsample_2(self.up_cnn_2(y2)))
        y0 = f(x0 + self.upsample_1(self.up_cnn_1(y1)))

        y = f(self.final_cnn(y0))

        plot_tensors()

        assert y.shape == (batch_size, 2, 65536)

        return y
