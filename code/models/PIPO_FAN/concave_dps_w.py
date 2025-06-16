import torch
import torch.nn as nn
import torch.nn.functional as F
from .concave_dps import ResUNet as ResUNet_0


class attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(attention, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
)
    def forward(self,x):
        x = self.conv(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.resnet = ResUNet_0(n_channels, n_classes)

        self.att = attention(n_classes, 1)


        self.gapool1 = nn.AvgPool3d(kernel_size=(96, 128, 128))
        self.gapool2 = nn.MaxPool3d(kernel_size=(96, 128, 128))

    def forward(self, x):
        a,b,c,d = self.resnet(x)
        
        w1 = self.att(a)
        w2 = self.att(b)
        w3 = self.att(c)
        w4 = self.att(d)        

        w1 = self.gapool1(w1) + self.gapool2(w1)
        w2 = self.gapool1(w2) + self.gapool2(w2)
        w3 = self.gapool1(w3) + self.gapool2(w3)
        w4 = self.gapool1(w4) + self.gapool2(w4)

        w = torch.cat((w1, w2, w3, w4), 1)
        reshaped_w= w.view(w.shape[0], w.shape[1], -1, w.shape[4])
        reshaped_w = torch.nn.Softmax2d()(reshaped_w)
        w = reshaped_w.view(w.shape[0], w.shape[1], w.shape[2], w.shape[3], w.shape[4])
        w1 = w[:,0:1,:,:,:]
        w2 = w[:,1:2,:,:,:]
        w3 = w[:,2:3,:,:,:]
        w4 = w[:,3:4,:,:,:]

        fi_out = w1*a + w2*b + w3*c + w4*d

        return fi_out