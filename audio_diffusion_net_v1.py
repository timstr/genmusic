import math
import torch
import torch.nn as nn

from torch_utils import CheckShape, Reshape

class AudioDiffusionNetV1(nn.Module):
    def __init__(self):
        super(AudioDiffusionNetV1, self).__init__()

        self.shapes = [
            (65536, 2),
            (16384, 8),
            (4096, 64),
            (1024, 128),
            (256, 256),
            (64, 256),
            # (32, 256),
            # (16, 256),
        ]

        # self.shapes = [
        #     (65536, 2),
        #     (16384, 8),
        #     (4096, 16),
        #     (1024, 32),
        #     (256, 64),
        #     (64, 128),
        #     (16, 128),
        # ]

        # self.shapes = [
        #     (65536, 2),
        #     (4096, 32),
        #     (256, 128),
        #     (16, 256),
        # ]

        # self.shapes = [
        #     (65536, 2),
        #     (4096, 32),
        #     (256, 128),
        # ]

        num_stages = len(self.shapes) - 1

        # kernel_size = 3
        # kernel_size = 5
        # kernel_size = 15
        kernel_size = 31
        # kernel_size = 127
        # kernel_size = 63

        padding_mode = "zeros"
        # padding_mode = "circular"

        activation_function = nn.LeakyReLU
        # activation_function = nn.Tanh
        # activation_function = nn.Tanh
        # activation_function = nn.ELU
        # activation_function = nn.Softplus
        # activation_function = nn.Tanhshrink
        # activation_function = lambda: UnaryModule(torch.sin)  # LOL
        # activation_function = lambda: UnaryModule(symmetric_log)

        time_embedding_hidden_features = 16
        self.temporal_features = time_embedding_hidden_features

        encoder = []
        decoder = []

        encoder_time_modules = []
        decoder_time_modules = []

        l_middle, c_middle = self.shapes[-1]

        self.middle = nn.Sequential(
            CheckShape((c_middle, l_middle)),
            nn.Conv1d(
                in_channels=c_middle,
                out_channels=c_middle,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                # padding_mode=padding_mode,
                padding_mode="zeros",
            ),
            nn.BatchNorm1d(num_features=c_middle),
            activation_function(),
            CheckShape((c_middle, l_middle)),
        )

        # self.middle_const = nn.parameter.Parameter(
        #     torch.rand((c_middle, l_middle)), requires_grad=True
        # )

        for i, ((l_prev, c_prev), (l_next, c_next)) in enumerate(
            zip(self.shapes, self.shapes[1:])
        ):
            assert (l_prev % l_next) == 0

            encoder_time_modules.append(
                nn.Sequential(
                    CheckShape((time_embedding_hidden_features,)),
                    nn.Linear(
                        in_features=time_embedding_hidden_features,
                        out_features=time_embedding_hidden_features,
                    ),
                    nn.BatchNorm1d(num_features=time_embedding_hidden_features),
                    activation_function(),
                    nn.Linear(
                        in_features=time_embedding_hidden_features, out_features=c_prev
                    ),
                    Reshape((c_prev,), (c_prev, 1)),
                )
            )

            decoder_time_modules.append(
                nn.Sequential(
                    CheckShape((time_embedding_hidden_features,)),
                    nn.Linear(
                        in_features=time_embedding_hidden_features,
                        out_features=time_embedding_hidden_features,
                    ),
                    nn.BatchNorm1d(num_features=time_embedding_hidden_features),
                    activation_function(),
                    nn.Linear(
                        in_features=time_embedding_hidden_features, out_features=c_next
                    ),
                    Reshape((c_next,), (c_next, 1)),
                )
            )

            # Convolution-based resampling
            e = nn.Sequential(
                CheckShape((c_prev, l_prev)),
                # CheckShape((c_prev, l_prev)),
                nn.Conv1d(
                    in_channels=c_prev,
                    # in_channels=c_prev,
                    out_channels=c_next,
                    kernel_size=kernel_size,
                    stride=(l_prev // l_next),
                    padding=(kernel_size - 1) // 2,
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm1d(num_features=c_next),
                activation_function(),
                CheckShape((c_next, l_next)),
                nn.Conv1d(
                    in_channels=c_next,
                    out_channels=c_next,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
            )

            # Resampling-based resizing
            # e = nn.Sequential(
            #     CheckShape((c_prev, l_prev)),
            #     Resample1d(new_length=l_next),
            #     CheckShape((c_prev, l_next)),
            #     nn.Conv1d(
            #         in_channels=c_prev,
            #         out_channels=c_next,
            #         kernel_size=kernel_size,
            #         stride=1,
            #         padding=(kernel_size - 1) // 2,
            #         padding_mode=padding_mode,
            #     ),
            #     nn.BatchNorm1d(num_features=c_next),
            #     nn.LeakyReLU(),
            #     # nn.Conv1d(
            #     #     in_channels=c_next,
            #     #     out_channels=c_next,
            #     #     kernel_size=kernel_size,
            #     #     stride=1,
            #     #     padding=(kernel_size - 1) // 2,
            #     #     padding_mode=padding_mode,
            #     # ),
            #     CheckShape((c_next, l_next)),
            # )

            encoder.append(e)

            # in_channels = c_next * (2 if i + 1 < num_stages else 1)
            # in_channels = c_next * 2
            in_channels = c_next

            # Convolution-based resizing
            d = nn.Sequential(
                CheckShape((in_channels, l_next)),
                # CheckShape((in_channels, l_next)),
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=c_prev,
                    kernel_size=kernel_size,
                    stride=(l_prev // l_next),
                    padding=(kernel_size - 1) // 2,
                    # output_padding=1,
                    output_padding=3,
                    # output_padding=15,
                ),
                nn.BatchNorm1d(num_features=c_prev),
                CheckShape((c_prev, l_prev)),
                activation_function(),
                nn.Conv1d(
                    in_channels=c_prev,
                    out_channels=c_prev,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    padding_mode=padding_mode,
                ),
            )

            # Resample-based resizing
            # d = nn.Sequential(
            #     CheckShape((in_channels, l_next)),
            #     nn.Conv1d(
            #         in_channels=in_channels,
            #         out_channels=c_prev,
            #         kernel_size=kernel_size,
            #         stride=1,
            #         padding=(kernel_size - 1) // 2,
            #         padding_mode=padding_mode,
            #     ),
            #     nn.BatchNorm1d(num_features=c_prev),
            #     nn.LeakyReLU(),
            #     # nn.Conv1d(
            #     #     in_channels=c_prev,
            #     #     out_channels=c_prev,
            #     #     kernel_size=kernel_size,
            #     #     stride=1,
            #     #     padding=(kernel_size - 1) // 2,
            #     #     padding_mode=padding_mode,
            #     # ),
            #     CheckShape((c_prev, l_next)),
            #     Resample1d(new_length=l_prev),
            #     CheckShape((c_prev, l_prev)),
            # )

            decoder.append(d)

        decoder.reverse()
        decoder_time_modules.reverse()

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        self.encoder_time_modules = nn.ModuleList(encoder_time_modules)
        self.decoder_time_modules = nn.ModuleList(decoder_time_modules)

        self.final = nn.Conv1d(
            in_channels=4,
            out_channels=2,
            kernel_size=1,
            stride=1,
            # padding=(kernel_size - 1) // 2,
            padding=0,
            padding_mode=padding_mode,
        )

    def forward(self, x, t, verbose=False):
        (batch_size,) = t.shape
        time_embedding = t.unsqueeze(-1).repeat(1, self.temporal_features)
        assert time_embedding.shape == (batch_size, self.temporal_features)
        freqs = (
            torch.linspace(
                start=1.0,
                end=(50.0 * 2.0 * math.pi),
                steps=(self.temporal_features // 2),
                device=t.device,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        assert freqs.shape == (batch_size, self.temporal_features // 2)
        time_embedding[:, : self.temporal_features // 2] = torch.sin(
            freqs * time_embedding[:, : self.temporal_features // 2]
        )
        time_embedding[:, self.temporal_features // 2 :] = torch.cos(
            freqs * time_embedding[:, self.temporal_features // 2 :]
        )
        time_embedding.requires_grad_(False)

        values = [x]

        # def cat_t(x):
        #     s = t.unsqueeze(-1).unsqueeze(-1)
        #     s = s * torch.ones_like(x[:, 0:1])
        #     return torch.cat([x, s], dim=1)

        for i, enc, in enumerate(self.encoder):
            if verbose:
                print(f"Encoder {i} : {x.shape[1:]}")

            # x = cat_t(x)
            x += self.encoder_time_modules[i](time_embedding)
            x = enc(x)

            if verbose:
                print(f"           -> {x.shape[1:]}")

            values.append(x)

        values.pop()

        x = self.middle(x)

        # x += self.middle_const

        for i, dec in enumerate(self.decoder):

            if verbose:
                print(f"Decoder {i} : {x.shape[1:]}")

            #     x += values.pop()
            # if i + 1 < len(self.decoder):

            x += self.decoder_time_modules[i](time_embedding)

            if i > 0:
                # x = torch.cat([x, values.pop()], dim=1)
                x += values.pop()

            # x = cat_t(x)

            x = dec(x)

            if verbose:
                print(f"           -> {x.shape[1:]}")

        x += values.pop()

        # x = self.final(torch.cat([x, values.pop()], dim=1))

        assert len(values) == 0

        return x


# # TEST
# model = AudioDiffusionNet()

# num_params = 0
# for p in model.parameters():
#     num_params += p.numel()
# print(f"The model has {num_params} parameters")

# x = torch.rand((4, 2, g_audio_length))
# t = torch.rand((4,))
# y = model(x, t, verbose=True)
# assert y.shape == (4, 2, g_audio_length)

# print("Ok!")
# exit(0)