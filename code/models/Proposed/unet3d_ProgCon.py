# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import interpolate

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        return self.dsv(input)

class ConvBlock(nn.Module):
    """
    Two 3D convolution layers with batch norm and leaky relu.
    Droput is used between the two convolution layers.
    
    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """
    3D downsampling followed by ConvBlock

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

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
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 trilinear=True):
        super(UpBlock, self).__init__()
        self.trilinear = trilinear
        if trilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.trilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D_ProgCon(nn.Module):
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
        super(UNet3D_ProgCon, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.trilinear = self.params['trilinear']
        assert(len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv= ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        if(len(self.ft_chns) == 5):
          self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
          self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
               dropout_p = self.dropout[3], trilinear=self.trilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 
               dropout_p = self.dropout[2], trilinear=self.trilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 
               dropout_p = self.dropout[1], trilinear=self.trilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 
               dropout_p = self.dropout[0], trilinear=self.trilinear) 
        
        # deep supervision
        self.dsv4 = UnetDsv3(
            in_size=self.ft_chns[3], out_size=self.n_class)
        self.dsv3 = UnetDsv3(
            in_size=self.ft_chns[2], out_size=self.n_class)
        self.dsv2 = UnetDsv3(
            in_size=self.ft_chns[1], out_size=self.n_class)
        self.dsv1 = nn.Conv3d(
            in_channels=self.ft_chns[0], out_channels=self.n_class, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        if(len(self.ft_chns) == 5):
          x4 = self.down4(x3)
          x_d3 = self.up1(x4, x3)
        else:
          x_d3 = x3
        x_d2 = self.up2(x_d3, x2)
        x_d1 = self.up3(x_d2, x1)
        x_d0 = self.up4(x_d1, x0)

        # Deep Supervision
        dsv4 = self.dsv4(x_d3)
        dsv3 = self.dsv3(x_d2)
        dsv2 = self.dsv2(x_d1)
        dsv1 = self.dsv1(x_d0)

        return dsv1, dsv2, dsv3, dsv4
