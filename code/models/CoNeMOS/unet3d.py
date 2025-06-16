from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from .film import FiLM
from torch.nn.functional import interpolate

class ConvBlock(nn.Module):
    """
    Two 3D convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.film_1 = FiLM(n_dims = 3)
        self.film_2 = FiLM(n_dims = 3)
        
    def forward(self, x, condition=None):
        x = self.conv_1(x)
        x = self.film_1(x, condition)
        x = self.conv_2(x)
        x = self.film_2(x, condition)
        
        return x

class DownBlock(nn.Module):
    """
    3D downsampling followed by ConvBlock

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    """
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, condition=None):
        x = self.maxpool(x)
        x = self.conv(x, condition)
        
        return x

class UpBlock(nn.Module):
    """
    3D upsampling followed by ConvBlock
    
    :param in_channels1: (int) Channel number of high-level features.
    :param in_channels2: (int) Channel number of low-level features.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param trilinear: (bool) Use trilinear for up-sampling (by default).
        If False, deconvolution is used for up-sampling. 
    """
    def __init__(self, in_channels1, in_channels2, out_channels,
                 trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels)

    def forward(self, x1, x2, condition=None):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, condition)
    

class UNet3D(nn.Module):
    """
    An implementation of the U-Net.
        
    * Reference: Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
      3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
      `MICCAI (2) 2016: 424-432. <https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49>`_
    
    Note that there are some modifications from the original paper, such as
    the use of batch normalization, dropout, leaky relu and deep supervision.

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 4 or 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param trilinear: (bool) Using trilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    :param multiscale_pred: (bool) Get multi-scale prediction.
    """
    def __init__(self, params):
        super(UNet3D, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.trilinear = self.params['trilinear']
        self.size_condition_vector = self.params['size_condition_vector']
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)
        
        self.condition_mlp = self.build_condition_mlp(self.size_condition_vector)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3])
        if(len(self.ft_chns) == 5):
          self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4])
          self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
               trilinear=self.trilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 
               trilinear=self.trilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 
               trilinear=self.trilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 
               trilinear=self.trilinear) 
    
        self.final_conv = nn.Conv3d(self.ft_chns[0], 1, kernel_size = 1)
    
    def build_condition_mlp(self, size_condition_vector):
        layers = []
        for _ in range(4):
            layers.append(nn.Linear(size_condition_vector, 64))
            layers.append(nn.LeakyReLU(0.2))
            size_condition_vector = 64
        return nn.Sequential(*layers)
       
    def forward(self, x, condition=None):
        condition = condition.to(torch.float32)
        condition = self.condition_mlp(condition)
        
        x0 = self.in_conv(x, condition)
        x1 = self.down1(x0, condition)
        x2 = self.down2(x1, condition)
        x3 = self.down3(x2, condition)
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3, condition)
          x_d3 = self.up1(x4, x3, condition)
        else:
          x_d3 = x3
        x_d2 = self.up2(x_d3, x2, condition)
        x_d1 = self.up3(x_d2, x1, condition)
        x_d0 = self.up4(x_d1, x0, condition)
        output = self.final_conv(x_d0)

        return output
