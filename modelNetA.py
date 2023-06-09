# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# File description: Realize the model definition function.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

__all__ = [
    "ResidualDenseBlock", "ResidualResidualDenseBlock",
    "Discriminator", "Generator",
    "DownSamplingNetwork"
]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
            channels (int): The number of channels in the input image.
            growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = out5 * 0.2 + identity

        return out



class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
            channels (int): The number of channels in the input image.
            growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = out5 * 0.2 + identity

        return out



class MiniResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
            channels (int): The number of channels in the input image.
            growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(MiniResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.leaky_relu(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = out5 * 0.2 + identity

        return out



class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growths)
        self.rdb2 = ResidualDenseBlock(channels, growths)
        self.rdb3 = ResidualDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = out * 0.2 + identity

        return out


class MiniResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growths (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growths: int) -> None:
        super(MiniResidualResidualDenseBlock, self).__init__()
        self.M_rdb1 = MiniResidualDenseBlock(channels, growths)
        self.M_rdb2 = MiniResidualDenseBlock(channels, growths)
        self.M_rdb3 = MiniResidualDenseBlock(channels, growths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.M_rdb1(x)
        out = self.M_rdb2(out)
        out = self.M_rdb3(out)
        out = out * 0.2 + identity
        return out



class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 512 x 512
            nn.Conv2d(2, 32, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 256 x 256
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 64 x 64
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 16 x 16
            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        #RLNet
        self.RLNetconv_block1 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))
        RLNettrunk = []
        for _ in range(4):
            RLNettrunk += [ResidualResidualDenseBlock(64, 32)]
        self.RLNettrunk = nn.Sequential(*RLNettrunk)
        self.RLNetconv_block2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.RLNetconv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.RLNetconv_block4 = nn.Sequential(
            nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1)),
            nn.Tanh()
        )

        #############################################################################
        #Generator
        self.conv_block1 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))

        trunk = []
        for _ in range(16):
            trunk += [ResidualResidualDenseBlock(64, 32)]
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv_block2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))


        # Upsampling convolutional layer.
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            #nn.Sigmoid()
        )

        self.conv_block0_branch0 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.Tanh()
        )

        self.conv_block0_branch1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),
            nn.Tanh()
        )

        self.conv_block1_branch0 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1)),
            #nn.LeakyReLU(0.2, True),
            #nn.Conv2d(32, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid()
        )



        self.conv_block1_branch1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid())




    def _forward_impl(self, x: Tensor) -> Tensor:
        #RLNet
        out1 = self.RLNetconv_block1(x)
        out = self.RLNettrunk(out1)
        out2 = self.RLNetconv_block2(out)
        out = out1 + out2
        out = self.RLNetconv_block3(out)
        out = self.RLNetconv_block4(out)
        rlNet_out = out + x

        #Generator
        out1 = self.conv_block1(rlNet_out)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="bicubic"))
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="bicubic"))
        out = self.conv_block3(out)
        #
        out = self.conv_block4(out)

        #demResidual = out[:, 1:2, :, :]
        #grayResidual = out[:, 0:1, :, :]

        # out = self.trunkRGB(out_4)
        #
        # out_dem = out[:, 3:4, :, :] * 0.2 + demResidual # DEM images extracted
        # out_rgb = out[:, 0:3, :, :] * 0.2 + rgbResidual # RGB images extracted

        #ra0
        #out_rgb=  rgbResidual + self.conv_block0_branch0(rgbResidual)

        out_dem = out + self.conv_block0_branch1(out) #out+ tanh()
        out_gray = out + self.conv_block0_branch0(out) #out+ tanh()

        out_gray = self.conv_block1_branch0(out_gray) #sigmoid()
        out_dem = self.conv_block1_branch1(out_dem) #sigmoid()

        return out_gray, out_dem, rlNet_out


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                m.weight.data *= 0.1
