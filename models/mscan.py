import torch.nn as nn
import torch.nn.functional as F
import torch


class SPPAttention(nn.Module):

    def __init__(self, channel, spp=(4, 2, 1)):
        super(SPPAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.c = channel
        self.spp = spp
        self.n = sum(spp)
        self.dense1 = nn.Linear(self.n * self.c, self.c)
        self.dense2 = nn.Linear(self.c, self.c // 16)
        self.dense3 = nn.Linear(self.c // 16, self.c)
        self.pool = dict()
        self.pool[1] = self.pool1_1
        self.pool[2] = self.pool1_2
        self.pool[4] = self.pool2_2
        self.pool[8] = self.pool2_4
        self.pool[16] = self.pool4_4

    def pool4_4(self, x):
        h = x.size(2)
        w = x.size(3)
        # 4 * 4
        image_v3_1_1 = self.global_pool(x[:, :, 0:int(h / 4), 0:int(w / 4)])
        image_v3_1_2 = self.global_pool(x[:, :, 0:int(h / 4), int(w / 4):int(w / 2)])
        image_v3_1_3 = self.global_pool(x[:, :, 0:int(h / 4), int(w / 2):int(w - w / 4)])
        image_v3_1_4 = self.global_pool(x[:, :, 0:int(h / 4), int(w - w / 4):w])

        image_v3_2_1 = self.global_pool(x[:, :, int(h / 4):int(h / 2), 0:int(w / 4)])
        image_v3_2_2 = self.global_pool(x[:, :, int(h / 4):int(h / 2), int(w / 4):int(w / 2)])
        image_v3_2_3 = self.global_pool(x[:, :, int(h / 4):int(h / 2), int(w / 2):int(w - w / 4)])
        image_v3_2_4 = self.global_pool(x[:, :, int(h / 4):int(h / 2), int(w - w / 4):w])

        image_v3_3_1 = self.global_pool(x[:, :, int(h / 2):int(h - h / 4), 0:int(w / 4)])
        image_v3_3_2 = self.global_pool(x[:, :, int(h / 2):int(h - h / 4), int(w / 4):int(w / 2)])
        image_v3_3_3 = self.global_pool(x[:, :, int(h / 2):int(h - h / 4), int(w / 2):int(w - w / 4)])
        image_v3_3_4 = self.global_pool(x[:, :, int(h / 2):int(h - h / 4), int(w - w / 4):w])

        image_v3_4_1 = self.global_pool(x[:, :, int(h - h / 4):h, 0:int(w / 4)])
        image_v3_4_2 = self.global_pool(x[:, :, int(h - h / 4):h, int(w / 4):int(w / 2)])
        image_v3_4_3 = self.global_pool(x[:, :, int(h - h / 4):h, int(w / 2):int(w - w / 4)])
        image_v3_4_4 = self.global_pool(x[:, :, int(h - h / 4):h, int(w - w / 4):w])
        image = torch.cat((image_v3_1_1, image_v3_1_2, image_v3_1_3, image_v3_1_4, image_v3_2_1,
                           image_v3_2_2, image_v3_2_3, image_v3_2_4, image_v3_3_1, image_v3_3_2, image_v3_3_3,
                           image_v3_3_4, image_v3_4_1, image_v3_4_2, image_v3_4_3, image_v3_4_4), 1)
        return image

    def pool2_2(self, x):
        h = x.size(2)
        w = x.size(3)
        image_v2_1_1 = self.global_pool(x[:, :, 0:int(h / 2), 0:int(w / 2)])
        image_v2_1_2 = self.global_pool(x[:, :, 0:int(h / 2), int(w / 2):w])
        image_v2_2_1 = self.global_pool(x[:, :, int(h / 2):h, 0:int(w / 2)])
        image_v2_2_2 = self.global_pool(x[:, :, int(h / 2):h, int(w / 2):w])
        image = torch.cat((image_v2_1_1, image_v2_1_2, image_v2_2_1, image_v2_2_2), 1)
        return image

    def pool2_4(self, x):
        h = x.size(2)
        w = x.size(3)
        image_1_1 = self.global_pool(x[:, :, 0:int(h/2), 0:int(w/4)])
        image_1_2 = self.global_pool(x[:, :, 0:int(h/2), int(w/4):int(w/2)])
        image_1_3 = self.global_pool(x[:, :, 0:int(h/2), int(w/2):int(w - w/4)])
        image_1_4 = self.global_pool(x[:, :, 0:int(h/2), int(w - w/4):w])

        image_2_1 = self.global_pool(x[:, :, int(h / 2): h, 0:int(w / 4)])
        image_2_2 = self.global_pool(x[:, :, int(h / 2): h, int(w / 4):int(w / 2)])
        image_2_3 = self.global_pool(x[:, :, int(h / 2): h, int(w / 2):int(w - w / 4)])
        image_2_4 = self.global_pool(x[:, :, int(h / 2): h, int(w - w / 4):w])
        image = torch.cat((image_1_1, image_1_2, image_1_3, image_1_4,
                           image_2_1, image_2_2, image_2_3, image_2_4), 1)
        return image

    def pool1_2(self, x):
        w = x.size(3)
        image_1_1 = self.global_pool(x[:, :, :, 0:int(w/2)])
        image_1_2 = self.global_pool(x[:, :, :, int(w/2):w])
        image = torch.cat((image_1_1, image_1_2), 1)
        return image

    def pool1_1(self, x):
        image = self.global_pool(x)
        return image

    def forward(self, x):
        spp = None
        for k in self.spp:
            t = self.pool[k](x)
            if spp is None:
                spp = t
            else:
                spp = torch.cat((spp, t), 1)
        spp = spp.view(-1, self.c * self.n)
        net = self.dense1(spp)
        act_net = F.relu(net)
        net = self.dense2(act_net)
        act_net = F.relu(net)
        net = self.dense3(act_net)
        act_net = torch.sigmoid(net)
        return act_net.view(-1, self.c, 1, 1)


class ResBlock(nn.Module):

    def __init__(self, channel, ksize=3, spp=(4, 2, 1)):
        super(ResBlock, self).__init__()
        padding = ksize // 2
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=ksize, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=ksize, padding=padding)
        self.spp_attention = SPPAttention(channel, spp)

    def forward(self, x):
        net = self.conv1(x)
        act_net = F.relu(net)
        net = self.conv2(act_net)
        scale = self.spp_attention(net)
        net = net * scale
        return x + net


class Encoder(nn.Module):

    def __init__(self, inchannel=3, ksize=3, spp=(4, 2, 1)):
        super(Encoder, self).__init__()
        # Conv1
        padding = ksize // 2
        self.layer1 = nn.Conv2d(inchannel, 32, kernel_size=ksize, padding=padding)
        self.layer2 = ResBlock(32, ksize=ksize, spp=spp)
        self.layer3 = ResBlock(32, ksize=ksize, spp=spp)
        self.layer4 = ResBlock(32, ksize=ksize, spp=spp)
        self.layer4p = ResBlock(32, ksize=ksize, spp=spp)
        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=ksize, stride=2, padding=padding)
        self.layer6 = ResBlock(64, ksize=ksize, spp=spp)
        self.layer7 = ResBlock(64, ksize=ksize, spp=spp)
        self.layer8 = ResBlock(64, ksize=ksize, spp=spp)
        self.layer8p = ResBlock(64, ksize=ksize, spp=spp)
        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=ksize, stride=2, padding=padding)
        self.layer10 = ResBlock(128, ksize=ksize, spp=spp)
        self.layer11 = ResBlock(128, ksize=ksize, spp=spp)
        self.layer12 = ResBlock(128, ksize=ksize, spp=spp)
        self.layer12p = ResBlock(128, ksize=ksize, spp=spp)

    def forward(self, x):
        # Conv1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        enc1_4 = self.layer4p(x)
        # Conv2
        x = self.layer5(enc1_4)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        enc2_4 = self.layer8p(x)
        # Conv3
        x = self.layer9(enc2_4)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer12p(x)
        return enc1_4, enc2_4, x


class Decoder(nn.Module):

    def __init__(self, outchannel=3, ksize=3, spp=(4, 2, 1)):
        super(Decoder, self).__init__()
        # Deconv3
        self.layer13p = ResBlock(128, ksize=ksize, spp=spp)
        self.layer13 = ResBlock(128, ksize=ksize, spp=spp)
        self.layer14 = ResBlock(128, ksize=ksize, spp=spp)
        self.layer15 = ResBlock(128, ksize=ksize, spp=spp)
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Deconv2
        self.layer17p = ResBlock(64, ksize=ksize, spp=spp)
        self.layer17 = ResBlock(64, ksize=ksize, spp=spp)
        self.layer18 = ResBlock(64, ksize=ksize, spp=spp)
        self.layer19 = ResBlock(64, ksize=ksize, spp=spp)
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Deconv1
        self.layer21p = ResBlock(32, ksize=ksize, spp=spp)
        self.layer21 = ResBlock(32, ksize=ksize, spp=spp)
        self.layer22 = ResBlock(32, ksize=ksize, spp=spp)
        self.layer23 = ResBlock(32, ksize=ksize, spp=spp)
        self.layer24 = nn.Conv2d(32, outchannel, kernel_size=ksize, padding=ksize // 2)

    def forward(self, enc1_4, enc2_4, x):
        # Deconv3
        x = self.layer13p(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        # Deconv2
        x = self.layer17p(x + enc2_4)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        # Deconv1
        x = self.layer21p(x + enc1_4)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        return x


class MSCAN(nn.Module):

    def __init__(self):
        super(MSCAN, self).__init__()
        self.encode1 = Encoder(ksize=3, spp=(1,))
        self.encode2 = Encoder(inchannel=6, ksize=3, spp=(2, 1))
        self.decode1 = Decoder(ksize=3, spp=(1,))
        self.decode2 = Decoder(ksize=3, spp=(2, 1))

    def encode_decode_level(self, x, last_scale_out, scale):
        '''
        two level in each scale
        :param x: input image
        :param last_scale_out: the output from last scale
        :return: the output of current scale
        '''
        enc1_4, enc2_4, feature2 = self.encode2(torch.cat([x, last_scale_out], dim=1))
        residual2 = self.decode2(enc1_4, enc2_4, feature2)
        enc1_4, enc2_4, tmp = self.encode1(x + residual2)
        feature1 = tmp + feature2
        y = self.decode1(enc1_4, enc2_4, feature1)
        return y

    def forward(self, x):
        output = []
        B3 = F.interpolate(x, scale_factor=1/4, mode='bilinear')
        I3 = self.encode_decode_level(B3, B3, 3)
        output.append(I3)
        I3 = I3.detach()
        B2 = F.interpolate(x, scale_factor=1/2, mode='bilinear')
        I3 = F.interpolate(I3, scale_factor=2, mode='bilinear')
        I2 = self.encode_decode_level(B2, I3, 2)
        output.append(I2)
        I2 = I2.detach()
        I2 = F.interpolate(I2, scale_factor=2, mode='bilinear')
        I1 = self.encode_decode_level(x, I2, 1)
        output.append(I1)
        return output

