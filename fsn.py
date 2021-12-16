import numpy as np
import os
import torch
from torch.nn import functional
import math
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import librosa
import soundfile as sf
from functools import partial
from pathlib import Path
from tqdm.notebook import tqdm
from torch.utils import data
from torch.utils.data import DataLoader


def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext

def drop_band(input, num_groups=2):
    """
    Reduce computational complexity of the sub-band part in the FullSubNet model.
    Shapes:
        input: [B, C, F, T]
        return: [B, C, F // num_groups, T]
    """
    batch_size, _, num_freqs, _ = input.shape
    assert batch_size > num_groups, f"Batch size = {batch_size}, num_groups = {num_groups}. The batch size should larger than the num_groups."

    if num_groups <= 1:
        # No demand for grouping
        return input

    # Each sample must has the same number of the frequencies for parallel training.
    # Therefore, we need to drop those remaining frequencies in the high frequency part.
    if num_freqs % num_groups != 0:
        input = input[..., :(num_freqs - (num_freqs % num_groups)), :]
        num_freqs = input.shape[2]

    output = []
    for group_idx in range(num_groups):
        samples_indices = torch.arange(group_idx, batch_size, num_groups, device=input.device)
        freqs_indices = torch.arange(group_idx, num_freqs, num_groups, device=input.device)

        selected_samples = torch.index_select(input, dim=0, index=samples_indices)
        selected = torch.index_select(selected_samples, dim=2, index=freqs_indices)  # [B, C, F // num_groups, T]

        output.append(selected)

    return torch.cat(output, dim=0)


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def unfold(input, num_neighbors):
        """
        Along the frequency axis, this function is used for splitting overlapped sub-band units.
        Args:
            input: four-dimension input.
            num_neighbors: number of neighbors in each side.
        Returns:
            Overlapped sub-band units.
        Shapes:
            input: [B, C, F, T]
            return: [B, N, C, F_s, T]. F_s represents the frequency axis of the sub-band unit, e.g. [2, 161, 1, 19, 200]
        """
        assert input.dim() == 4, f"The dim of the input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbors < 1:  # No change on the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)
        # N + 1(특정 frequency 자신) + N 
        sub_band_unit_size = num_neighbors * 2 + 1

        # 스펙토그램의 아래, 위를 반사해서 덧붙임
        output = functional.pad(output, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B * C, 1, F, T]
        
        # [B * C, F_s * T, F(=N)]
        # kernel size : (sub band, time step)
        # kernel이 freq 방향으로 슬라이딩 하면서 F & T 병합
        output = functional.unfold(output, (sub_band_unit_size, num_frames))  # move on the F and T axes.
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dim of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, F_s, T]
        # 앞뒤로 2N+1개 만큼의 주파수를 sub band로 묶는다
        # sub band의 step개수는 기존 주파수의 개수와 동일하다
        return output

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """
        Group Dropout for FullSubNet
        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:
        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 应该被 3 整除，否则最后一部分 batch 内的频率无法很好的训练
        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def forgetting_norm(input, sample_length=192):
        """
        Using the mean value of the near frames to normalization
        Args:
            input: feature
            sample_length: length of the training sample, used for calculating smooth factor
        Returns:
            normed feature
        Shapes:
            input: [B, C, F, T]
            sample_length_in_training: 192
        """
        assert input.ndim == 4
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

        eps = 1e-10
        mu = 0
        alpha = (sample_length - 1) / (sample_length + 1)

        mu_list = []
        # TODO bugs
        # TODO 是否需要修改为包含F的计算方法？
        for frame_idx in range(num_frames):
            if frame_idx < sample_length:
                alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        output = input / (mu + eps)

        output = output.reshape(batch_size, num_channels, num_freqs, num_frames)
        return output

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:
        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = 1e-10

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input):
        """
        Args:
            input: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        # utterance-level mu
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = input / (mu + 1e-5)

        return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """
        Args:
            input: [B, C, F, T]
        Returns:
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.std(input, dim=(1, 2, 3), keepdim=True)

        normed = (input - mu) / (std + 1e-5)

        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm
        Args:
            input: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum) / entry_count + cumulative_mean.pow(2)  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        elif norm_type == "forgetting_norm":
            norm = self.forgetting_norm
        else:
            raise NotImplementedError("You must set up a type of Norm. "
                                      "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc.")
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

class SequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        Wrapper of conventional sequence models (LSTM or GRU)
        Args:
            input_size : 프레임당 입력 크기
            output_size: when projection_size> 0, the linear layer is used for projection. Otherwise, no linear layer.
            hidden_size: 시퀀스 모델의 히든 유닛 수
            num_layers:  레이어 수 
            bidirectional: 양방향 유무
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

        self.output_activate_function = output_activate_function
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        assert x.dim() == 3, f"The shape of input is {x.shape}."
        self.sequence_model.flatten_parameters()

        # contiguous 메모리에서 요소를 연속적으로 만드는 것은 모델 최적화에 도움이 되지만 새 공간이 할당됩니다
        # 네트워크가 많은 계산을 시작하기 전에 사용하는 것이 좋습니다.
        x = x.permute(0, 2, 1)  # [B, F, T] => [B, T, F]
        o, _ = self.sequence_model(x)

        if self.output_size:
            o = self.fc_output_layer(o)

        if self.output_activate_function:
            o = self.activate_function(o)
        o = o.permute(0, 2, 1)  # [B, T, F] => [B, F, T]
        return o


class Model(BaseModel):
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
                 ):
        """
        FullSubNet model (cIRM mask)
        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        # pad 용
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram
        Returns:
            The real part and imag part of the enhanced spectrogram
        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        # 
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold fullband model's output, [B, N=F, C, F_f, T]. N is the number of sub-band units
        # full-band? output unfold해도 변화 없음.
        # 주파수 1개씩 묶고(sub band 크기), 기존 주파수 step만큼 움직이는데 무슨 변화가 있겠음
        fb_output_unfolded = self.unfold(fb_output, num_neighbors=self.fb_num_neighbors) # fb_num_neigh = 0
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy spectrogram, [B, N=F, C, F_s, T]
        # if sb_num_neigh = 15,
        # sub-band? output을 31개씩 묶고(sub band 크기), 기존 주파수 step 만큼 움직임
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbors=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, (F_s + F_f), T] = [B, F, 32, T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation.
        # These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()
        
        # pad 한만큼 앞부분도 마찬가지로 skip한 뒤, 0부분 제외하고 출력
        # 이렇게 하면 input은 t0, t1, ..., 인데 mask는 t2, t3 ..., 이런식으로 연산됨
        # 이건 실시간으로 노이즈 제거하기 위해서, input관련 연산 때문에 delay 준거 ㅇㅇㅇ
        output = sb_mask[:, :, :, self.look_ahead:]
        return output


def decompress_cIRM(mask, K=10, limit=9.9):
    """
    Uncompress cIRM from [-K ~ K] to [-inf, +inf]
    Args:
        mask: cIRM mask
        K: default 10
        limit: default 0.1
    References:
        https://ieeexplore.ieee.org/document/7364200
    """
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask


def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists(), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def stft(y, n_fft, hop_length, win_length):
    """
    Wrapper of the official torch.stft for single-channel and multi-channel
    Args:
        y: single- or multi-channel speech with shape of [B, C, T] or [B, T]
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
    Shapes:
        mag: [B, F, T] if dims of input is [B, T], whereas [B, C, F, T] if dims of input is [B, C, T]
    Returns:
        mag, phase, real and imag with the same shape of [B, F, T] (**complex-valued** STFT coefficients)
    """
    num_dims = y.dim()
    assert num_dims == 2 or num_dims == 3, "Only support 2D or 3D Input"

    batch_size = y.shape[0]
    num_samples = y.shape[-1]

    if num_dims == 3:
        y = y.reshape(-1, num_samples)

    complex_stft = torch.stft(y, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=y.device),
                              return_complex=True)
    _, num_freqs, num_frames = complex_stft.shape

    if num_dims == 3:
        complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)

    mag, phase = torch.abs(complex_stft), torch.angle(complex_stft)
    real, imag = complex_stft.real, complex_stft.imag
    return mag, phase, real, imag


def istft(features, n_fft, hop_length, win_length, length=None, input_type="complex"):
    """
    Wrapper of the official torch.istft
    Args:
        features: [B, F, T] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft: num of FFT
        hop_length: hop length
        win_length: hanning window size
        length: expected length of istft
        use_mag_phase: use mag and phase as the input ("features")
    Returns:
        single-channel speech of shape [B, T]
    """
    if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
        assert isinstance(features, tuple) or isinstance(features, list)
        real, imag = features
        features = torch.complex(real, imag)
    elif input_type == "complex":
        assert isinstance(features, torch.ComplexType)
    elif input_type == "mag_phase":
        # the feature is (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    else:
        raise NotImplementedError("Only 'real_imag', 'complex', and 'mag_phase' are supported")

    return torch.istft(features, n_fft, hop_length, win_length, window=torch.hann_window(n_fft, device=features.device),
                       length=length)


def prepare_device(n_gpu: int, keep_reproducibility=False):
    """
    Choose to use CPU or GPU depend on the value of "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment. if n_gpu == 0, use CPU; if n_gpu >= 1, use GPU.
        keep_reproducibility (bool): if we need to consider the repeatability of experiment, set keep_reproducibility to True.
    See Also
        Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        # possibly at the cost of reduced performance
        if keep_reproducibility:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.benchmark = False  # ensures that CUDA selects the same convolution algorithm each time
            torch.set_deterministic(True)  # configures PyTorch only to use deterministic implementation
        else:
            # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda:0")

    return device


class Inferencer:
    def __init__(self, dataset_dir_list, checkpoint_path, output_dir):
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        root_dir = Path(output_dir).expanduser().absolute()
        self.device = prepare_device(torch.cuda.device_count())

        print("Loading inference dataset...")
        self.dataloader = self._load_dataloader(dataset_dir_list)
        print("Loading model...")
        self.model, epoch = self._load_model(checkpoint_path, self.device)

        self.enhanced_dir = root_dir / f"enhanced_{str(epoch).zfill(4)}"
        self.noisy_dir = root_dir / f"noisy"

        # self.enhanced_dir = root_dir
        prepare_empty_dir([self.noisy_dir, self.enhanced_dir])

        # Supported STFT
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        self.sr = 16000

        # See utils_backup.py
        self.torch_stft = partial(stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.torch_istft = partial(istft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.librosa_stft = partial(librosa.stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        self.librosa_istft = partial(librosa.istft, hop_length=self.hop_length, win_length=self.win_length)
        
    @staticmethod
    def _load_model(checkpoint_path, device):
        model = Model(num_freqs=257, look_ahead=2, sequence_model="LSTM",
        fb_num_neighbors=0, sb_num_neighbors=15, fb_output_activate_function="ReLU",
        sb_output_activate_function=False, fb_model_hidden_size=512,
        sb_model_hidden_size=384, weight_init=False, norm_type="offline_laplace_norm",
        num_groups_in_drop_band=2)

        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        print(f"Loading model checkpoint (epoch == {epoch})...")

        model_static_dict = {key.replace("module.", ""): value for key, value in model_static_dict.items()}
        model.load_state_dict(model_static_dict)
        model.to(device)
        model.eval()
        return model, model_checkpoint["epoch"]

    @staticmethod
    def _load_dataloader(dataset_dir_list, sr=16000):
        dataset = Dataset(dataset_dir_list, sr)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
        )
        return dataloader

    @torch.no_grad()
    def full_band_crm_mask(self, noisy, inference_args):
        noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)

        noisy_mag = noisy_mag.unsqueeze(1)
        pred_crm = self.model(noisy_mag)
        pred_crm = pred_crm.permute(0, 2, 3, 1)

        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_real - pred_crm[..., 1] * noisy_imag
        enhanced_imag = pred_crm[..., 1] * noisy_real + pred_crm[..., 0] * noisy_imag
        enhanced = self.torch_istft((enhanced_real, enhanced_imag), length=noisy.size(-1), input_type="real_imag")
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()
        return enhanced

    @torch.no_grad()
    def __call__(self):
        inference_type = "full_band_crm_mask"
        assert inference_type == "full_band_crm_mask", f"Not implemented Inferencer type: {inference_type}"

        inference_args = 15

        for noisy, name in tqdm(self.dataloader, desc="Inference"):
            assert len(name) == 1, "The batch size of inference stage must 1."
            name = name[0]

            enhanced = getattr(self, inference_type)(noisy.to(self.device), inference_args)

            if abs(enhanced).any() > 1:
                print(f"Warning: enhanced is not in the range [-1, 1], {name}")

            amp = np.iinfo(np.int16).max
            enhanced = np.int16(0.8 * amp * enhanced / np.max(np.abs(enhanced)))
            sf.write(self.enhanced_dir / f"{name}.wav", enhanced, samplerate=16000)

            noisy = noisy.detach().squeeze(0).numpy()
            if np.ndim(noisy) > 1:
                noisy = noisy[0, :]  # first channel
            noisy = noisy[:enhanced.shape[-1]]
            sf.write(self.noisy_dir / f"{name}.wav", noisy, samplerate=16000)


class BaseDataset(data.Dataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _offset_and_limit(dataset_list, offset, limit):
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]
        return dataset_list

    @staticmethod
    def _parse_snr_range(snr_range):
        assert len(snr_range) == 2, f"The range of SNR should be [low, high], not {snr_range}."
        assert snr_range[0] <= snr_range[-1], f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list


class Dataset(BaseDataset):
    def __init__(self,
                 dataset_dir_list,
                 sr,
                 ):
        """
        Args:
            noisy_dataset_dir_list (str or list): noisy dir or noisy dir list
        """
        super().__init__()
        assert isinstance(dataset_dir_list, list)
        self.sr = sr

        noisy_file_path_list = []
        for dataset_dir in dataset_dir_list:
            dataset_dir = Path(dataset_dir).expanduser().absolute()
            noisy_file_path_list += librosa.util.find_files(dataset_dir.as_posix())  # Sorted

        self.noisy_file_path_list = noisy_file_path_list
        self.length = len(self.noisy_file_path_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_file_path = self.noisy_file_path_list[item]
        noisy_y = librosa.load(noisy_file_path, sr=self.sr)[0]
        noisy_y = noisy_y.astype(np.float32)

        return noisy_y, basename(noisy_file_path)[0]


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NEG_INF = torch.finfo(torch.float32).min
    PI = math.pi
    SOUND_SPEED = 343  # m/s
    EPSILON = np.finfo(np.float32).eps
    MAX_INT16 = np.iinfo(np.int16).max
    checkpoint_path = '~/Downloads/fullsubnet_best_model_58epochs.tar'
    output_dir = '~/Downloads'
    dataset_dir_list = ["~/Downloads/Test/Audio/",]

    with torch.no_grad():
        model = Model(
            num_freqs=257,
            look_ahead=2,
            sequence_model="LSTM",
            fb_num_neighbors=0,
            sb_num_neighbors=15,
            fb_output_activate_function="ReLU",
            sb_output_activate_function=False,
            fb_model_hidden_size=512,
            sb_model_hidden_size=384,
            norm_type="offline_laplace_norm",
            num_groups_in_drop_band=2,
            weight_init=False,
        )
        infer = Inferencer(dataset_dir_list, checkpoint_path, output_dir)
        infer()
    print('fin')