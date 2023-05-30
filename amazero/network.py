import torch
import torch.nn as nn


class AmazonsNetwork(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        residual_layers: int,
        board_shape: tuple[int, int]
    ):
        super().__init__()
        self.board_shape = board_shape
        in_channels = 4
        board_size = board_shape[0] * board_shape[1]
        policy_size = 3 * board_size
        self.conv_input = ConvolutionBlock(in_channels, residual_channels, conv3x3)
        self.res_tower = ResidualTower(residual_channels, residual_layers)
        self.head = Head(residual_channels, board_size, policy_size)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv_input(x)
        out = self.res_tower(out)
        pol, val = self.head(out)
        pol = pol.reshape((batch_size, 3) + self.board_shape)
        return pol, val


class MultistageResidualNetwork(nn.Module):
    def __init__(
            self,
        in_channels: tuple[int, ...],
        in_channel_indices: tuple[torch.IntTensor],
        residual_layers: tuple[int, ...],
        residual_channels: int,
        out_size: int,
    ):
        super().__init__()
        stage_count = len(in_channels)
        assert stage_count == len(residual_layers)
        assert stage_count == len(in_channel_indices)
        self.in_channel_indices = in_channel_indices
        self.stages = nn.ModuleList([ResidualNetwork(
            in_channels=in_channels[i] + (residual_channels if i != 0 else 0),
            residual_channels=residual_channels,
            residual_layers=residual_layers[i],
            out_size=out_size
        ) for i in range(stage_count)])
    
    def forward(self, x: tuple[torch.Tensor, ...]):
        assert len(x) == len(self.stages)
        headout = ()
        mixin = []
        for i, stage in enumerate(self.stages):
            features = [t.index_select(dim=1, index=self.in_channel_indices[i]) for t in x[i:]]
            batch = torch.cat(features, dim=0)
            xselect = torch.cat([batch] + mixin, dim=1)
            pol, val, stage_towerout = stage(xselect)
            next_stage_offset = x[i].size(dim=0)
            headout += ((pol[:next_stage_offset], val[:next_stage_offset]),)
            mixin = [stage_towerout[next_stage_offset:]]
        return headout


class ResidualNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
        input_size: int,
        policy_size: int
    ):
        super().__init__()
        self.conv_input = ConvolutionBlock(in_channels, residual_channels, conv3x3)
        self.res_tower = ResidualTower(residual_channels, residual_layers)
        self.head = Head(residual_channels, input_size, policy_size)

    def forward(self, x):
        out = self.conv_input(x)
        out = self.res_tower(out)
        out = self.head(out)
        return out


class ResidualTower(nn.Sequential):
    def __init__(self, residual_channels: int, residual_layers: int):
        super().__init__(*[
                ResidualBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
        ])


class Head(nn.Module):
    def __init__(self, residual_channels: int, input_size: int, policy_size: int):
        super().__init__()
        self.policy_head = PolicyHead(residual_channels, input_size, policy_size)
        self.value_head = ValueHead(residual_channels, input_size)

    def forward(self, x):
        pol = self.policy_head(x)
        val = self.value_head(x)
        return pol, val


class PolicyHead(nn.Module):
    def __init__(self, residual_channels: int, input_size: int, out_size: int):
        super().__init__()
        self.policy_conv = ConvolutionBlock(residual_channels, 6, conv1x1)
        self.policy_fc = nn.Linear(6 * input_size, out_size)

    def forward(self, x):
        out = self.policy_conv(x)
        out = self.policy_fc(torch.flatten(out, start_dim=1))
        return out


class ValueHead(nn.Module):
    def __init__(self, residual_channels: int, input_size: int):
        super().__init__()
        self.value_conv = ConvolutionBlock(residual_channels, 1, conv1x1)
        self.value_fc_1 = nn.Linear(input_size, 256)
        self.value_fc_2 = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.value_conv(x)
        out = self.relu(self.value_fc_1(torch.flatten(out, start_dim=1)))
        out = torch.tanh(self.value_fc_2(out))
        return out


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, convolution, relu=True):
        super().__init__()
        self.conv = convolution(in_channels, out_channels)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvolutionBlock(in_channels, out_channels, conv3x3)
        self.conv2 =  ConvolutionBlock(in_channels, out_channels, conv3x3, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


def conv3x3(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

device = torch.device("mps")

BATCH = 64
BOARD_DIM = 10

# model = MultistageResidualNetwork(
#     in_channels=(4, 6, 6),
#     in_channel_indices=(
#         # -- stage 0
#         # 0 = friendly amazons
#         # 1 = enemy amazons
#         # 2 = arrows
#         # 3 = empty spaces
#         # -- stage 1
#         # 4 = unselected friendly amazons
#         # 5 = selected friendly amazon
#         # 6 = selected friendly amazon legal moves
#         # -- stage 2
#         # 7 = friendly amazon selected move
#         # 8 = friendly amazons post move
#         # 9 = empty spaces post move
#         # 10 = legal arrows moves
#         torch.IntTensor([0, 1, 2, 3]).to(device),
#         torch.IntTensor([1, 2, 3, 4, 5, 6]).to(device),
#         torch.IntTensor([1, 2, 7, 8, 9, 10]).to(device),
#     ),
#     residual_channels=256,
#     residual_layers=(12, 4, 4),
#     out_size=BOARD_DIM*BOARD_DIM
# ).to(device)

model = AmazonsNetwork(
    residual_channels=256,
    residual_layers=20,
    board_shape=(BOARD_DIM, BOARD_DIM)
).to(device)

test_batch = torch.randint(0, 2, (BATCH, 4, BOARD_DIM, BOARD_DIM), dtype=torch.float).to(device)

import time
N = 8
t1 = time.time()
results = []
for i in range(N):
    with torch.no_grad():
        pol, val = model(test_batch)
    print(pol.shape)
    results.append(pol.to("cpu"))
    results.append(val.to("cpu"))
# torch.cuda.synchronize(device)
t2 = time.time()
print((t2-t1)/N)
