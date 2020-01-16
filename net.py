from torch import nn
from config import args
import torch


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_size = args.board_size

        # 输入为[n,2,15,15]
        self.conv_layer = nn.Sequential(
            ConvolutionLayer(2, 64, 3, 1, 1),  # [n,64,15,15]
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(64),
            ConvolutionLayer(64, 128, 3, 1, 1),  # [n,128,15,15]
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
            ResidualLayer(128),
            ConvolutionLayer(128, 256, 3, 1, 1),  # [n,256,15,15]
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ConvolutionLayer(256, 16, 1),  # [n,16,15,15]
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
            ResidualLayer(16),
        )

        self.linear_action_layer = nn.Sequential(
            nn.Linear(16 * self.board_size * self.board_size, 256),
            # nn.BatchNorm1d(256),
            nn.PReLU(),
        )

        self.linear_value_layer = nn.Sequential(
            nn.Linear(16 * self.board_size * self.board_size, 256),
            # nn.BatchNorm1d(256),
            nn.PReLU(),
        )

        self.action_layer = nn.Linear(256, self.board_size * self.board_size)
        self.value_layer = nn.Linear(256, 1)

        self.softmax = nn.LogSoftmax(-1)
        self.tanh = nn.Tanh()

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.reshape(data.size(0), -1)
        action = self.linear_action_layer(data)
        action = self.softmax(self.action_layer(action))
        value = self.linear_value_layer(data)
        value = self.tanh(self.value_layer(value))
        return action, value

    def add_histogram(self, writer, epoch):
        writer.add_histogram("weight/action", self.action_layer.weight, epoch)
        writer.add_histogram("weight/value", self.value_layer.weight, epoch)


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.PReLU()
        )

    def forward(self, data):
        return self.conv_layer(data)


class ResidualLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvolutionLayer(input_channels, input_channels // 2, 1, 1, 0),
            ConvolutionLayer(input_channels // 2, input_channels // 2, 3, 1, 1),
            ConvolutionLayer(input_channels // 2, input_channels, 1, 1, 0)
        )

    def forward(self, data):
        return data + self.conv(data)


if __name__ == '__main__':
    input = torch.Tensor(2, 2, args.board_size, args.board_size)
    net = MyNet()
    action, value = net(input)
    print(action.shape)
    print(value.shape)
    params = sum([param.numel() for param in net.parameters()])
    print(params)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from config import args
#
#
# class MyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.board_size = args.board_size
#
#         # 综合网络
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#
#         # 策略网络，输出动作概率的对数 (方便后续计算)
#         self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
#         self.act_fc1 = nn.Linear(4 * self.board_size * self.board_size, self.board_size * self.board_size)
#
#         # 估值网络，输出当前局面价值 (归一化到-1 到 1之间）
#         self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
#         self.val_fc1 = nn.Linear(2 * self.board_size * self.board_size, 64)
#         self.val_fc2 = nn.Linear(64, 1)
#
#     def forward(self, state_input):
#         # common layers
#         x = F.relu(self.conv1(state_input))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#
#         # action policy layers
#         x_act = F.relu(self.act_conv1(x))
#         x_act = x_act.view(-1, 4 * self.board_size * self.board_size)
#         x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
#
#         # state value layers
#         x_val = F.relu(self.val_conv1(x))
#         x_val = x_val.view(-1, 2 * self.board_size * self.board_size)
#         x_val = F.relu(self.val_fc1(x_val))
#         x_val = torch.tanh(self.val_fc2(x_val))
#
#         return x_act, x_val
#
#
# if __name__ == '__main__':
#     input = torch.Tensor(2, 2, 9, 9)
#     net = Net(9)
#     action, value = net(input)
#     print(action.shape)
#     print(value.shape)
#     params = sum([param.numel() for param in net.parameters()])
#     print(params)
