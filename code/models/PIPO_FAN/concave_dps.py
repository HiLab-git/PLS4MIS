import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    
class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(one_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class res_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_conv, self).__init__()
        self.conv1 = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        if x.shape == x1.shape:
            r = x + x1
        else:
            r = self.bridge(x) + x1
        return r


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.pool = nn.MaxPool3d(2)
        self.mpconv = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)

    def forward(self, x, y):
        x = self.pool(x)
        # Concatenation
        x_1 = torch.cat((x,y),1)
        # Summation
        # x_1 = x + y
        x_2 = self.mpconv(x_1)
        if x_1.shape == x_2.shape:
            x = x_1 + x_2
        else:
            x = self.bridge(x_1) + x_2
        return x
    

class up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, kernel_size=2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.bridge = one_conv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        diffZ = x1.size()[4] - x2.size()[4]
        x2 = F.pad(x2, (diffZ // 2, int(diffZ / 2),
                        diffY // 2, int(diffY / 2),
                        diffX // 2, int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x) + self.bridge(x)
        return x

    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    
class ResUNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.dbconv1 = res_conv(16, 32)
        self.down1 = down(32, 32)
        self.dbconv2 = res_conv(16,32)
        self.dbconv3 = res_conv(32,64)
        self.down2 = down(64, 64)
        self.down3 = down(128, 64)
        self.dbup1 = res_conv(64,32)
        self.dbup2 = res_conv(32,16)
        self.dbup3 = res_conv(16,16)
        self.up1 = up(128, 32)
        self.dbup4 = res_conv(32,16)
        self.dbup5 = res_conv(16,16)
        self.up2 = up(64, 16)
        self.dbup6 = res_conv(16,16)
        self.up3 = up(32, 16)
        self.outc1 = outconv(16, n_classes)
        self.outc2 = outconv(16, n_classes)
        self.outc3 = outconv(16, n_classes)
        self.outc = outconv(16, n_classes)
        self.pool = nn.AvgPool3d(2)
        self.unpool = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        y1 = self.pool(x)
        z1 = self.inc(y1)
        x2 = self.down1(x1, z1)
        y2 = self.pool(y1)
        z2 = self.inc(y2)
        a1 = self.dbconv1(z2)
        x3 = self.down2(x2, a1)
        y3 = self.pool(y2)
        z3 = self.inc(y3)
        a2 = self.dbconv2(z3)
        a3 = self.dbconv3(a2)
        x4 = self.down3(x3, a3)
        o1 = self.dbup1(x4)
        o1 = self.dbup2(o1)
        o1 = self.dbup3(o1)
        out1 = self.outc1(o1)
        x5 = self.up1(x4, x3)
        o2 = self.dbup4(x5)
        o2 = self.dbup5(o2)
        out2 = self.outc2(o2)
        x6 = self.up2(x5, x2)
        o3 = self.dbup6(x6)
        out3 = self.outc3(o3)
        o4 = self.up3(x6, x1)
        out4 = self.outc(o4)
        
        out1 = self.unpool(self.unpool(self.unpool(out1)))
        out2 = self.unpool(self.unpool(out2))
        out3 = self.unpool(out3)

        return out1, out2, out3, out4
