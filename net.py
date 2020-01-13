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
            ConvolutionLayer(64, 128, 3, 1, 1),  # [n,128,15,15]
            ConvolutionLayer(128, 256, 3, 1, 1),  # [n,256,15,15]
            ConvolutionLayer(256, 16, 1)  # [n,16,15,15]
        )

        self.linear_action_layer = nn.Sequential(
            nn.Linear(16 * self.board_size * self.board_size, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
        )

        self.linear_value_layer = nn.Sequential(
            nn.Linear(16 * self.board_size * self.board_size, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
        )

        self.action_layer = nn.Linear(256, 1)
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


if __name__ == '__main__':
    input = torch.Tensor(2, 2, 15, 15)
    net = MyNet()
    action, value = net(input)
    print(action.shape)
    print(value.shape)
    params = sum([param.numel() for param in net.parameters()])
    print(params)
