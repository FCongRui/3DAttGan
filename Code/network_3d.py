import torch
import torch.nn as nn
import torch.nn.functional as F


def make_3d_network():
    return Network_3d()


def make_Discriminator():
    return discriminator()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(16, 16 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(16 // 16, 16, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # y1= self.avg_pool(x)
        # y2 = self.fc1(y1)
        # y3 = self.relu1(y2)
        # avg_out = self.fc2(y3)
        #
        # y11= self.max_pool(x)
        # y22 = self.fc1(y11)
        # y33 = self.relu1(y22)
        # max_out = self.fc2(y33)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Network_3d(nn.Module):

    def __init__(self, downsample=None):
        super(Network_3d, self).__init__()

        planes =16
        self.K, S, P, OP = (3, [2, 4, 4], 1, [ 0, 3, 3])

        self.base = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=planes, kernel_size= 3,padding=1),
            nn.PReLU(),
        )
        self.next = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size= 3,padding=1),
            nn.PReLU(),
        )


        self.r = nn.PReLU()

        self.last = nn.Conv3d(16, 3, kernel_size=3, padding=1)


        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        # self.downsample = downsample
        # self.stride = stride

        self.upFrame = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=self.K, stride=S, padding=P, output_padding=OP),
            nn.PReLU(),
            nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        a = 3
        # b, c, n, h, w = x.shape
        old = x
        # old = x[:, 1:2, :, :]
        ##Block1
        out = self.base(x)
        out = self.next(out)

        for i in range(a):
            out = self.next(out)
            out = self.next(out)
            out = self.ca(out) * out
            out = self.sa(out) * out

            out = self.r(out)

        out = self.upFrame(out)


        # b, c, n, h, w = x.shape
        # out_Ay = torch.zeros(b, c ,n,h,w).cuda()
        # for i in range(c):
        #     x = out[:, i, :, :, :]
        #     y = self.ca(x) * x
        #     Ay = self.sa(y) * y
        #     out_Ay[:, i, :, :, :] = Ay



        # if self.downsample is not None:
        #     residual = self.downsample(x)


        ##Block2
        # out = self.base1(out)
        #
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        #
        # out = self.r(out)


        # out = self.last(out)
        # out = out +old
        # out += residual
        # out = self.relu(out)

        return out

def swish(x):
    return x * F.sigmoid(x)

class Discriminator_Feature(nn.Module):
    def __init__(self):
        super(Discriminator_Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)



        # Replaced original paper FC layers with FCN
        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = swish(self.conv1(x))

        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

class Discriminator_Motion(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(Discriminator_Motion, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,0.02 / n)  # this modified initialization seems to work better, but it's very hacky
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, output_more=False):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        # we don't have the gt for flow, we just fine tune it on flownets
        if not output_more:
            return flow2
        else:
            return [flow2, flow3, flow4, flow5, flow6]
        # if self.training:
        #     return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return flow2 ，

def add_sn(m):
        for name, layer in m.named_children():
            m.add_module(name, add_sn(layer))
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            return nn.utils.spectral_norm(m)
        else:
            return m


def discriminator():
    a = Discriminator_Feature()
    b = Discriminator_Motion()
    c = a + b
    my_model = add_sn(c)
    return my_model



