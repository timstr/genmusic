import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import math
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import random

from loss_plotter import LossPlotter
from torch_utils import CircularPad1d, save_module, restore_module
from audio_util import load_audio_clips, random_audio_batch
from signal_processing import make_spectrogram
from progress_bar import progress_bar
from util import plt_screenshot

from audio_diffusion_net_v1 import AudioDiffusionNetV1
from audio_diffusion_net_v2 import AudioDiffusionNetV2

class UnaryModule(nn.Module):
    def __init__(self, function):
        super(UnaryModule, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


g_audio_length = 65536
# g_audio_length = 16384

def symmetric_log(x):
    sign = torch.sign(x)
    return sign * torch.log(torch.abs(x + sign))


device = torch.device("cuda")

# adapted from https://www.youtube.com/watch?v=a4Yfz2FxXiY&t=772s&ab_channel=DeepFindr

g_num_time_steps = 100

g_noise_schedule = torch.linspace(start=0.0001, end=0.1, steps=g_num_time_steps)
g_cumulative_noise_schedule = 1.0 - torch.cumprod(1.0 - g_noise_schedule, dim=0)

if (1.0 - g_cumulative_noise_schedule[-1]) > 0.01:
    print("WARNING: final cumulative noise value is not close to one")

# g_betas = torch.linspace(start=0.0001, end=0.2, steps=g_num_time_steps)
# g_alphas = 1.0 - g_betas
# g_alphas_cumprod = torch.cumprod(g_alphas, dim=0)
# if g_alphas_cumprod[-1] > 0.01:
#     print("WARNING: final alpha cumprod value is not close to zero")
# g_alphas_cumprod_prev = nn.functional.pad(g_alphas_cumprod[:-1], (1, 0), value=1.0)
# g_sqrt_recip_alphas = torch.sqrt(1.0 / g_alphas)
# g_sqrt_alphas_cumprod = torch.sqrt(g_alphas_cumprod)
# g_sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - g_alphas_cumprod)
# g_posterior_variance = (
#     g_betas * (1.0 - g_alphas_cumprod_prev) / (1.0 - g_alphas_cumprod)
# )

# import matplotlib.pyplot as plt
# plt.plot(g_noise_schedule, c="r")
# plt.plot(g_cumulative_noise_schedule, c="k")
# plt.show()
# exit(0)

def forward_diffusion_sample(x_0):
    batch_size = x_0.shape[0]

    time_steps_int = [random.randrange(g_num_time_steps) for _ in range(batch_size)]

    time_steps_float = torch.tensor(
        data=[t / (g_num_time_steps - 1) for t in time_steps_int],
        dtype=torch.float32,
        device=x_0.device,
        requires_grad=False,
    )

    cumulative_noise_values = torch.tensor(
        data=[g_cumulative_noise_schedule[i] for i in time_steps_int],
        dtype=torch.float32,
        device=x_0.device,
        requires_grad=False,
    )

    stddev_scale = cumulative_noise_values
    mean_scale = torch.sqrt(1.0 - stddev_scale)

    while mean_scale.ndim < x_0.ndim:
        mean_scale = mean_scale.unsqueeze(-1)
        stddev_scale = stddev_scale.unsqueeze(-1)

    noise = torch.randn_like(x_0)

    x_t = (x_0 * mean_scale) + (noise * stddev_scale)

    return x_t, noise, time_steps_float



    # sqrt_alphas_cumprod_t = torch.zeros((batch_size,))
    # sqrt_one_minus_alphas_cumprod_t = torch.zeros((batch_size,))
    # for i in range(batch_size):
    #     sqrt_alphas_cumprod_t[i] = g_sqrt_alphas_cumprod[t[i]]
    #     sqrt_one_minus_alphas_cumprod_t[i] = g_sqrt_one_minus_alphas_cumprod[t[i]]

    # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None]
    # sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None]

    # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(device=device)
    # sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(device=device)

    # sqrt_alphas_cumprod_t.requires_grad_(False)
    # sqrt_one_minus_alphas_cumprod_t.requires_grad_(False)

    # noise = torch.randn_like(x_0)
    # noise.requires_grad_(False)

    # return (
    #     ((sqrt_alphas_cumprod_t * x_0) + (sqrt_one_minus_alphas_cumprod_t * noise)),
    #     noise,
    # )


@torch.no_grad()
def sample_timestep(model, x, t):
    batch_size = x.shape[0]
    assert x.shape == (batch_size, 2, g_audio_length)
    assert isinstance(t, int)

    # betas_t = g_betas[t].item()
    # sqrt_one_minus_alphas_cumprod_t = g_sqrt_one_minus_alphas_cumprod[t].item()
    # sqrt_recip_alphas_t = g_sqrt_recip_alphas[t].item()
    # # sigma_t = math.sqrt(g_posterior_variance[t].item())
    # sigma_t = 0.1 * betas_t
    # # sigma_t = 0.1 * betas_t
    t_0_to_1 = t / (g_num_time_steps - 1)

    # betas_t = torch.full((batch_size, 1, 1), fill_value=betas_t, device=device)
    # sqrt_one_minus_alphas_cumprod_t = torch.full(
    #     (batch_size, 1, 1), fill_value=sqrt_one_minus_alphas_cumprod_t, device=device
    # )
    # sqrt_recip_alphas_t = torch.full(
    #     (batch_size, 1, 1), fill_value=sqrt_recip_alphas_t, device=device
    # )
    # sigma_t = torch.full((batch_size, 1, 1), fill_value=sigma_t, device=device)
    t_0_to_1 = torch.full((batch_size,), fill_value=t_0_to_1, device=device)

    model_prediction = model(x, t_0_to_1)

    # HACK normalizing model prediction
    # pred_var, pred_mean = torch.var_mean(model_prediction, dim=2, keepdim=True)
    # model_prediction = (model_prediction - pred_mean) / torch.sqrt(pred_var.clamp(min=1.0))

    # if predicting the signal
    clean_prediction, predicted_noise = model_prediction, (x - model_prediction)

    # if predicting the noise
    # clean_prediction, predicted_noise = (x - model_prediction), model_prediction

    if t == 0:
        x_predicted = clean_prediction
    else:
        denoise_amount = 0.5
        renoise_amount = 0.1
        noise = torch.randn_like(x)
        stddev_scale = g_cumulative_noise_schedule[t]
        # mean_scale = math.sqrt(1.0 - stddev_scale)

        denoised = (denoise_amount * clean_prediction) + ((1.0 - denoise_amount) * x)

        renoised = denoised + (renoise_amount * stddev_scale * noise)

        x_predicted = renoised

    assert x_predicted.shape == (batch_size, 2, g_audio_length)
    return x_predicted, predicted_noise


@torch.no_grad()
def sample(model, batch_size, steps_per_intermediate=None, gather_noise_stats=False):
    model.eval()
    x = torch.randn((batch_size, 2, g_audio_length), device=device)

    intermediates = []
    pred_noise_stats = []
    sample_stats = []

    for t in range(0, g_num_time_steps)[::-1]:
        # print(t)
        x, pred_noise = sample_timestep(model=model, x=x, t=t)
        if gather_noise_stats:
            pred_min = torch.min(pred_noise)
            pred_var, pred_mean = torch.var_mean(pred_noise)
            pred_max = torch.max(pred_noise)

            pred_min = pred_min.item()
            pred_stddev = math.sqrt(pred_var.item())
            pred_mean = pred_mean.item()
            pred_max = pred_max.item()
            pred_noise_stats.append(
                [
                    pred_min,
                    pred_mean - pred_stddev,
                    pred_mean,
                    pred_mean + pred_stddev,
                    pred_max,
                ]
            )

            x_min = torch.min(x)
            x_var, x_mean = torch.var_mean(x)
            x_max = torch.max(x)

            x_min = x_min.item()
            x_stddev = math.sqrt(x_var.item())
            x_mean = x_mean.item()
            x_max = x_max.item()
            sample_stats.append(
                [
                    x_min,
                    x_mean - x_stddev,
                    x_mean,
                    x_mean + x_stddev,
                    x_max,
                ]
            )

        # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # x = torch.clamp(x, min=-10.0, max=10.0)
        if steps_per_intermediate and (t % steps_per_intermediate == 0):
            intermediates.append(x)

    if steps_per_intermediate:
        return torch.stack(intermediates, dim=1), pred_noise_stats, sample_stats
    else:
        return x, pred_noise_stats, sample_stats


def train(
    model,
    optimizer,
    batch,
    plot_activations,
):
    model.train()
    batch_size = batch.shape[0]

    # t = torch.rand((batch_size,), device=device)
    # sigmas = sigma_start + t * (sigma_end - sigma_start)
    # s = torch.exp(sigmas)
    # noise = s.reshape(batch_size, 1, 1) * torch.randn_like(batch, device=device)
    # noise.requires_grad_(False)
    # noisy_batch = batch + noise

    # t_0_to_1 = torch.zeros((batch_size,))
    # t = []
    # for i in range(batch_size):
    #     t_i = random.randrange(0, g_num_time_steps)
    #     t.append(t_i)
    #     t_0_to_1[i] = t_i / (g_num_time_steps - 1)
    # t_0_to_1 = t_0_to_1.to(device=device)

    # Hmmm it appears that by NOT scaling the noise here, the loss function gives
    # equal treatment to both high and low variances found at different ends of
    # the diffusion process.
    noisy_batch, noise, t = forward_diffusion_sample(x_0=batch)

    prediction = model(noisy_batch, t, plot_activations=plot_activations)

    # if predicting the signal
    error = prediction - batch

    # if predicting the noise
    # error = prediction - noise

    loss_wrt_t = torch.mean(
        torch.mean(torch.square(error), dim=-1), dim=-1
    )
    assert loss_wrt_t.shape == (batch_size,)
    loss = torch.mean(loss_wrt_t)
    # loss = torch.mean(torch.abs(predicted_noise - noise))
    optimizer.zero_grad()
    loss.backward()

    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.000000000001)
    nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1) # Hmmm this makes an interesting difference

    optimizer.step()
    return loss.item(), t.detach().cpu(), loss_wrt_t.detach().cpu()


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.convs = nn.Sequential(
            CircularPad1d(1),
            nn.Conv1d(
                in_channels=2,
                out_channels=2,
                kernel_size=3,
                stride=1,
                # padding=1,
                # padding_mode="circular",
            ),
        )

    def forward(self, x, t):
        return self.convs(x)


def main():
    all_audio_clips = load_audio_clips("input_sound/*.flac")

    # HACK
    # all_audio_clips = load_audio_clips("input_sound/all u ever want 3.flac")

    if g_audio_length != 65536:
        all_resampled_clips = []
        with torch.no_grad():
            for i, (clip, clip_name) in enumerate(all_audio_clips):
                resampled_clip = nn.functional.interpolate(
                    clip.unsqueeze(0), size=g_audio_length,
                ).squeeze(0)
                assert resampled_clip.shape == (2, g_audio_length)
                all_resampled_clips.append((resampled_clip, clip_name))
                progress_bar(i, len(all_audio_clips), "clips resampled")
        all_audio_clips = all_resampled_clips

    # HACK
    # all_audio_clips = all_audio_clips[:1]
    # all_audio_clips = all_audio_clips[417:418]
    # HACKKK
    # all_audio_clips[0][0][...] *= 0.0

    # HACKKKKK
    # all_audio_clips_tensor = torch.cat(
    #     [audio for audio, _name in all_audio_clips], dim=1
    # )
    # assert all_audio_clips_tensor.ndim == 2
    # assert all_audio_clips_tensor.shape[0] == 2
    # audio_var, audio_mean = torch.var_mean(all_audio_clips_tensor, dim=1)
    # print(f"audio_var = {audio_var}")
    # print(f"audio_mean = {audio_mean}")
    # exit(0)
    dataset_mean = 0.0
    # dataset_stddev = 0.1
    dataset_stddev = 1.0

    # HACKKKK
    # model = DummyModel().to(device=device)
    # model = AudioDiffusionNetV1().to(device=device)
    model = AudioDiffusionNetV2().to(device=device)

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(f"The model has {num_params} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # torch.set_anomaly_enabled(True)

    # Chosen by bypassing model and using perfect noise predictions
    # Produces a smooth result over time assuming an exponential noise schedule
    # sigma_start = 0
    # sigma_end = -10

    # batch_size = 128
    # batch_size = 32
    # batch_size = 16
    batch_size = 4

    pickup_iteration = None
    # pickup_iteration = 4459

    plt.ion()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

    ax_tl = axes[0, 0]
    ax_tm = axes[0, 1]
    ax_tr = axes[0, 2]
    ax_trr = axes[0, 3]
    ax_bl = axes[1, 0]
    ax_bm = axes[1, 1]
    ax_br = axes[1, 2]
    ax_brr = axes[1, 3]

    plot_interval = 512
    loss_interval = 128
    plot_quantiles = 8
    sounds_per_plot = 4

    loss_plotter = LossPlotter(loss_interval, (0.0, 0.0, 0.8), plot_quantiles)

    loss_vs_time_plotter_0 = LossPlotter(
        loss_interval * 4, (0.0, 0.0, 1.0), plot_quantiles
    )
    loss_vs_time_plotter_1 = LossPlotter(
        loss_interval * 4, (0.0, 0.33, 0.66), plot_quantiles
    )
    loss_vs_time_plotter_2 = LossPlotter(
        loss_interval * 4, (0.0, 0.66, 0.33), plot_quantiles
    )
    loss_vs_time_plotter_3 = LossPlotter(
        loss_interval * 4, (0.0, 1.0, 0.0), plot_quantiles
    )

    current_iteration = 0 if pickup_iteration is None else pickup_iteration

    def save_things(iteration):
        save_module(
            model, f"models/model_{iteration + 1}.dat",
        )
        save_module(
            optimizer, f"models/model_optimizer_{iteration + 1}.dat",
        )
        loss_plotter.save(f"models/loss_plotter_{iteration + 1}.dat")
        loss_vs_time_plotter_0.save(
            f"models/loss_vs_time_plotter_0_{iteration + 1}.dat"
        )
        loss_vs_time_plotter_1.save(
            f"models/loss_vs_time_plotter_1_{iteration + 1}.dat"
        )
        loss_vs_time_plotter_2.save(
            f"models/loss_vs_time_plotter_2_{iteration + 1}.dat"
        )
        loss_vs_time_plotter_3.save(
            f"models/loss_vs_time_plotter_3_{iteration + 1}.dat"
        )

    if pickup_iteration is not None:
        restore_module(model, f"models/model_{pickup_iteration}.dat")
        restore_module(optimizer, f"models/model_optimizer_{pickup_iteration}.dat")
        loss_plotter.load(f"models/loss_plotter_{pickup_iteration}.dat")
        loss_vs_time_plotter_0.load(
            f"models/loss_vs_time_plotter_0_{pickup_iteration}.dat"
        )
        loss_vs_time_plotter_1.load(
            f"models/loss_vs_time_plotter_1_{pickup_iteration}.dat"
        )
        loss_vs_time_plotter_2.load(
            f"models/loss_vs_time_plotter_2_{pickup_iteration}.dat"
        )
        loss_vs_time_plotter_3.load(
            f"models/loss_vs_time_plotter_3_{pickup_iteration}.dat"
        )

    try:
        for _ in range(1_000_000):

            time_to_plot = (
                (current_iteration + 1) % plot_interval
            ) == 0 or current_iteration <= (
                1 + (0 if pickup_iteration is None else pickup_iteration)
            )

            with torch.no_grad():
                batch = random_audio_batch(
                    batch_size=batch_size,
                    patch_size=g_audio_length,
                    audio_clips=all_audio_clips,
                )

            # normalize
            batch = (batch - dataset_mean) / dataset_stddev

            loss, t_batch, loss_wrt_t_batch = train(
                model=model,
                optimizer=optimizer,
                batch=batch,
                # sigma_start=sigma_start,
                # sigma_end=sigma_end,
                plot_activations=time_to_plot,
            )

            for t, l in zip(t_batch, loss_wrt_t_batch):
                t = t.item()
                l = l.item()
                if t < 0.25:
                    loss_vs_time_plotter_0.append(l)
                elif t < 0.5:
                    loss_vs_time_plotter_1.append(l)
                elif t < 0.75:
                    loss_vs_time_plotter_2.append(l)
                else:
                    loss_vs_time_plotter_3.append(l)

            loss_plotter.append(loss)

            progress_bar(
                current_iteration % plot_interval,
                plot_interval,
                f"total: {current_iteration + 1}",
            )

            if time_to_plot:
                with torch.no_grad():
                    output_clips, noise_stats, sample_stats = sample(
                        model=model,
                        batch_size=sounds_per_plot,
                        gather_noise_stats=True
                        # steps=100,
                        # sigma_start=sigma_start,
                        # sigma_end=sigma_end,
                    )
                    assert output_clips.shape == (sounds_per_plot, 2, g_audio_length,)
                    output_clips = (output_clips * dataset_stddev) + dataset_mean
                    output_clips = output_clips.cpu()

                    intermediates_batch, _, _= sample(
                        model=model,
                        batch_size=3,
                        # steps=100,
                        # sigma_start=sigma_start,
                        # sigma_end=sigma_end,
                        steps_per_intermediate=(g_num_time_steps // 10),
                        gather_noise_stats=False,
                    )
                    assert intermediates_batch.shape == (3, 10, 2, g_audio_length)
                    intermediates_batch = (intermediates_batch * dataset_stddev) + dataset_mean
                    intermediates_batch = intermediates_batch.cpu()

                audio_clip, song_name = random.choice(all_audio_clips)

                ax_tl.cla()
                ax_tm.cla()
                ax_tr.cla()
                ax_trr.cla()
                ax_bl.cla()
                ax_bm.cla()
                ax_br.cla()
                ax_brr.cla()

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

                ax_tr.title.set_text("Loss")
                ax_tr.set_yscale("log")
                loss_plotter.plot_to(ax_tr)

                spectrogram_stacks = []
                for intermediates in intermediates_batch:
                    intermediate_spectrograms = [
                        make_spectrogram(c[0:1]) for c in intermediates
                    ]

                    spectrogram_stacks.append(
                        torch.cat(intermediate_spectrograms, dim=0)
                    )
                ax_br.title.set_text("Intermediate Samples")
                ax_br.imshow(
                    torch.cat(spectrogram_stacks, dim=1).unsqueeze(2).repeat(1, 1, 3)
                )

                ax_trr.title.set_text("Loss vs. time (0=blue, T=green)")
                ax_trr.set_yscale("log")
                loss_vs_time_plotter_0.plot_to(ax_trr)
                loss_vs_time_plotter_1.plot_to(ax_trr)
                loss_vs_time_plotter_2.plot_to(ax_trr)
                loss_vs_time_plotter_3.plot_to(ax_trr)

                ax_brr.title.set_text("Predicted noise vs. time (0=blue, T=green)")
                ax_brr.set_ylim(-1.5, 1.5)
                for t, (stats_prev, stats_next) in enumerate(
                    zip(noise_stats, noise_stats[1:])
                ):
                    p0, p1, p2, p3, p4 = stats_prev
                    n0, n1, n2, n3, n4 = stats_next
                    tf = t / (len(noise_stats) + 1)
                    rgb = (0.0, 1.0 - tf, tf)
                    ax_brr.fill_between(
                        x=[t, t + 1], y1=[p0, n0], y2=[p4, n4], color=(rgb + (0.25,))
                    )
                    ax_brr.fill_between(
                        x=[t, t + 1], y1=[p1, n1], y2=[p3, n3], color=(rgb + (0.5,))
                    )
                    ax_brr.plot(
                        [t, t + 1],
                        [p2, n2],
                        color=(0.0, tf * 0.5, 0.5 - 0.5 * tf, 1.0,),
                    )
                ax_brr.plot(
                    [0, g_num_time_steps], [0, 0], "k--",
                )
                ax_brr.plot(g_cumulative_noise_schedule.flip(0), "k--")
                ax_brr.plot(-g_cumulative_noise_schedule.flip(0), "k--")
                ax_brr.plot(g_noise_schedule.flip(0), "r--")
                ax_brr.plot(-g_noise_schedule.flip(0), "r--")
                sample_q0s, sample_q1s, sample_q2s, sample_q3s, sample_q4s = zip(*sample_stats)
                ax_brr.plot(sample_q0s, color=(0.0, 0.0, 0.0, 0.5), linewidth=1)
                ax_brr.plot(sample_q1s, color=(0.0, 0.0, 0.0, 0.5), linewidth=2)
                ax_brr.plot(sample_q2s, color=(0.0, 0.0, 0.0, 0.5), linewidth=4)
                ax_brr.plot(sample_q3s, color=(0.0, 0.0, 0.0, 0.5), linewidth=2)
                ax_brr.plot(sample_q4s, color=(0.0, 0.0, 0.0, 0.5), linewidth=1)

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
                        sample_rate=(32000 * g_audio_length // 65536),
                    )

            if ((current_iteration + 1) % 8192) == 0:
                save_things(current_iteration)

            current_iteration += 1

    except KeyboardInterrupt as e:
        print("\n\nControl-C detected, saving model...\n")
        save_things(current_iteration)
        print("Exiting")
        exit(1)


if __name__ == "__main__":
    main()
