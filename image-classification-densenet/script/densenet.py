import math
import os
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


class _TransitionLayer(nn.Sequential):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__(
            OrderedDict(
                [
                    ('bn', nn.BatchNorm2d(num_input_channels)),
                    ('relu', nn.ReLU(inplace=True)),
                    ('conv', nn.Conv2d(num_input_channels, num_output_channels,
                                       kernel_size=1, stride=1, bias=False)),
                    ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                ]
            )
        )


class _DenseLayer(nn.Module):
    def __init__(self, num_input_channels, num_output_channels,
                 multiply_factor, dropout_rate):
        super().__init__()
        num_conv1_output_channels = multiply_factor * num_output_channels
        self.dropout_rate = dropout_rate
        self.part1 = nn.Sequential(
            OrderedDict(
                [
                    ('bn1', nn.BatchNorm2d(num_input_channels)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('conv1', nn.Conv2d(num_input_channels, num_conv1_output_channels,
                                        kernel_size=1, stride=1, bias=False))
                ]
            )
        )
        self.part2 = nn.Sequential(
            OrderedDict(
                [
                    ('bn2', nn.BatchNorm2d(num_conv1_output_channels)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('conv2', nn.Conv2d(num_conv1_output_channels, num_output_channels,
                                        kernel_size=3, stride=1, padding=1))
                ]
            )
        )

    def forward(self, input_features):
        output = self.part1(input_features)
        output = self.part2(output)
        if self.dropout_rate > 0:
            output = F.dropout2d(output, p=self.dropout_rate, training=self.training)
        return output


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_channels, growth_rate, multiply_factor,
                 dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for index in range(num_layers):
            self.layers.append(_DenseLayer(num_input_channels + index * growth_rate, growth_rate,
                                           multiply_factor, dropout_rate))

    def forward(self, input_features):
        new_features = input_features
        for layer in self.layers:
            output = layer(new_features)
            new_features = torch.cat((new_features, output), dim=1)
        return new_features


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_channels=24, multiply_factor=4, dropout_rate=0,
                 num_classes=10, memory_efficient=False, save_path=None):
        super().__init__()
        self.memory_efficient = memory_efficient
        self.main_net = nn.Sequential(
            OrderedDict(
                [
                    ('conv0', nn.Conv2d(3, num_init_channels, kernel_size=7,
                                        stride=2, padding=3)),
                    ('bn0', nn.BatchNorm2d(num_init_channels)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                ]
            )
        )
        num_input_channels = num_init_channels
        for i, block_depth in enumerate(block_config):
            self.main_net.add_module('block{}'.format(i + 1),
                                     _DenseBlock(block_depth, num_input_channels, growth_rate,
                                                 multiply_factor, dropout_rate))
            num_input_channels += growth_rate * block_depth
            if i == len(block_config) - 1:
                continue
            self.main_net.add_module('transition{}'.format(i + 1),
                                     _TransitionLayer(num_input_channels,
                                                      int(num_input_channels * compression)))
            num_input_channels = int(num_input_channels * compression)
        self.main_net.add_module('bnfinal', nn.BatchNorm2d(num_input_channels))
        self.main_net.add_module('relufinal', nn.ReLU(inplace=True))
        self.classifier = nn.Linear(num_input_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(save_path, 'model.pth'))

    def forward(self, input_features):
        if self.memory_efficient:
            output = cp.checkpoint_sequential(self.main_net, 4, input_features)
        else:
            output = self.main_net(input_features)
        output = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
        output = self.classifier(output)
        return output


if __name__ == '__main__':
    model = fire.Fire(DenseNet)
    # for i, m in model.named_parameters():
    #     print(i, m)

