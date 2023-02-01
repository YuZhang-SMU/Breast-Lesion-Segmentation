import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.triple_conv(x)

class SingleConv1(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size=3, padding=1):
        super().__init__()
        self.Single_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.Single_Conv(x)

class SingleConv2(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size=3, padding=1, dila_rate=1):
        super().__init__()
        self.Single_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ker_size, padding=padding, dilation=dila_rate),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.Single_Conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = TripleConv(in_channels, out_channels)
        self.maxp = nn.MaxPool2d(kernel_size=2)

    def forward(self, input_f):
        conv_f = self.conv(input_f)
        max_f = self.maxp(conv_f)
        return conv_f, max_f

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv =TripleConv(in_channels, out_channels)

    def forward(self, decode_feature, encode_feature):
        x = self.ups(decode_feature)
        x = torch.cat([x, encode_feature], dim=1)
        x = self.conv(x)
        return x

class encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(2,64)
        self.down2 = Down(64,128)
        self.down3 = Down(128,256)
        self.down4 = Down(256,512)
        self.conv = TripleConv(512,1024)

    def forward(self, x):
        e1,max1 = self.down1(x)
        e2,max2 = self.down2(max1)
        e3,max3 = self.down3(max2)
        e4,max4 = self.down4(max3)
        e5 = self.conv(max4)
        return [e4,e3,e2,e1],e5

class decode(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = Up(1536,512)
        self.up2 = Up(768,256)
        self.up3 = Up(384,128)
        self.up4=  Up(192,64)

    def forward(self,x, e4, e3, e2, e1):
        d1 = self.up1(x, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        return [d1, d2, d3, d4]

class BAF_unit(nn.Module):
    def __init__(self, in_channels_m, in_channels_ms, out_channels, frag=False):
        super(BAF_unit, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_f = SingleConv2(in_channels_m, out_channels, ker_size=1, padding=0, dila_rate=2)
        if frag:
            self.conv1_b = SingleConv2(128, out_channels, ker_size=1, padding=0)
        else:
            self.conv1_b = SingleConv2(in_channels_m, out_channels, ker_size=1, padding=0)
        self.conv2 = SingleConv2(in_channels_ms, out_channels//2, ker_size=1, padding=0)
        self.conv3 = SingleConv1(out_channels, out_channels, ker_size=1, padding=0)
        self.conv4 = SingleConv1(out_channels, out_channels, ker_size=3)
        self.conv5 = SingleConv1(out_channels, out_channels//8, ker_size=1, padding=0)
        self.conv6 = SingleConv1(out_channels+out_channels//8, 128, ker_size=1, padding=0)
        self.conv7 = SingleConv1(out_channels, out_channels//2, ker_size=3)
        self.conv8 = SingleConv1(out_channels//2+128, out_channels // 4, ker_size=3)

    def forward(self, fore_f, back_f):
        # back_path
        up_back = self.ups(back_f[1])
        conv_dila_b = self.conv1_b(up_back)
        conv_be = self.conv2(back_f[0])
        conv_bd = self.conv2(back_f[2])
        conca_b1 = torch.cat([conv_be, conv_bd], dim=1)
        add_b = conv_dila_b+conca_b1

        act_b = self.act(add_b)
        conv_b1 = self.conv3(act_b)
        conv_b2 = self.conv4(conv_b1)
        conv_b3 = self.conv5(conv_b2)
        conca_b2 = torch.cat([add_b, conv_b3], dim=1)
        conv_b4 = self.conv6(conca_b2)
        conv_b4_r = conv_b4.reshape(conv_b4.size()[0], conv_b4.size()[1], conv_b4.size()[2]*conv_b4.size()[3])
        max_value, max_index = torch.max(conv_b4_r, dim=2, keepdim=True)
        max_c = max_value.reshape(conv_b4.size()[0], conv_b4.size()[1], 1, 1)
        out_back = max_c-conv_b4

        # fore_path
        up_fore = self.ups(fore_f[1])
        conv_dila_f = self.conv1_f(up_fore)
        conv_fe = self.conv2(fore_f[0])
        conv_fd = self.conv2(fore_f[2])
        conca_f1 = torch.cat([conv_fe, conv_fd], dim=1)
        add_f = conv_dila_f + conca_f1
        act_f = self.act(add_f)
        conv_f1 = self.conv3(act_f)
        conv_f2 = self.conv7(conv_f1)
        conca_f2 = torch.cat([conv_f2, out_back], dim=1)
        conv_f3 = self.conv8(conca_f2)
        conca_f3 = torch.cat([add_f, conv_f3], dim=1)

        return conca_f3, conv_b4

class Shape_aware_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shape_aware_unit, self).__init__()
        self.conv1 = SingleConv1(in_channels, out_channels, ker_size=1, padding=0)
        self.conv2 = SingleConv1(out_channels, out_channels//2, ker_size=3)
        self.conv3 = SingleConv1(out_channels//2, out_channels // 4, ker_size=3)
        self.conv4 = SingleConv1(out_channels//2, 16, ker_size=1, padding=0)

    def forward(self, input_f):
        conv1 = self.conv1(input_f)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conca = torch.cat([input_f, conv3], dim=1)
        out_shape = self.conv4(conv2)
        return conca, out_shape

class Edge_aware_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Edge_aware_unit, self).__init__()
        self.conv1 = SingleConv1(in_channels, out_channels, ker_size=1, padding=0)
        self.conv2 = SingleConv1(out_channels, out_channels // 2, ker_size=3)
        self.conv3 = SingleConv1(out_channels//2, out_channels //4, ker_size=3)

    def detect_edge(self, inputs, sobel_kernel):
        kernel = np.array(sobel_kernel, dtype='float32')
        kernel = kernel.reshape((1, 1, 3, 3))
        weight = Variable(torch.from_numpy(kernel)).to(device)
        edge = torch.zeros(inputs.size()[1],inputs.size()[0],inputs.size()[2],inputs.size()[3]).to(device)
        for k in range(inputs.size()[1]):
            fea_input = inputs[:,k,:,:]
            fea_input = fea_input.unsqueeze(1)
            edge_c = F.conv2d(fea_input, weight, padding=1)
            edge[k] = edge_c.squeeze(1)
        edge = edge.permute(1, 0, 2, 3)
        return edge

    def sobel_conv2d(self, inputs):
        edge_detect1 = self.detect_edge(inputs, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_detect2 = self.detect_edge(inputs, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        edge_detect3 = self.detect_edge(inputs, [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
        edge_detect4 = self.detect_edge(inputs, [[2, 1, 0], [1, 0, -1], [0, -1, -2]])
        edge = edge_detect1+edge_detect2+edge_detect3+edge_detect4
        return edge

    def forward(self, input_f):
        conv1 = self.conv1(input_f)
        conv2 = self.conv2(conv1)
        edge_f = self.sobel_conv2d(conv2)
        conv3 = self.conv3(edge_f)
        conca = torch.cat([input_f, conv3], dim=1)
        return conca

class Position_aware_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Position_aware_unit, self).__init__()
        self.conv1 = SingleConv1(in_channels, out_channels, ker_size=1, padding=0)
        self.conv2 = SingleConv1(out_channels, out_channels // 2, ker_size=3)
        self.conv3 = SingleConv1(out_channels // 2+1, out_channels // 4, ker_size=3)

    def forward(self, input_f, input_pos):
        conv1 = self.conv1(input_f)
        conv2 = self.conv2(conv1)
        conca1 = torch.cat([conv2, input_pos], dim=1)
        conv3 = self.conv3(conca1)
        conca = torch.cat([input_f, conv3], dim=1)
        return conca

class Middle_stream_block(nn.Module):
    def __init__(self, in_channels_m, in_channels_ms, out_channels, frag=False):
        super(Middle_stream_block, self).__init__()
        self.BAF = BAF_unit(in_channels_m, in_channels_ms, out_channels, frag=frag)
        self.shape = Shape_aware_unit(out_channels+out_channels//4, out_channels)
        self.edge = Edge_aware_unit(out_channels+out_channels//2, out_channels)
        self.position = Position_aware_unit(out_channels+out_channels//2+out_channels//4, out_channels)
    def forward(self, fore_e, back_e, fore_d, back_d, fore_m, back_m, input_pos):
        out_fore, out_back = self.BAF([fore_e, fore_m, fore_d], [back_e, back_m, back_d])
        out_shape1, out_shape2 = self.shape(out_fore)
        out_edge = self.edge(out_shape1)
        out_pos = self.position(out_edge, input_pos)
        return out_pos, out_back, out_shape2

class Middle_stream(nn.Module):
    def __init__(self):
        super(Middle_stream, self).__init__()
        self.ms_block1 = Middle_stream_block(1024, 512, 256, frag=False)
        self.ms_block2 = Middle_stream_block(512, 256, 128, frag=True)
        self.ms_block3 = Middle_stream_block(256, 128, 64, frag=True)
        self.ms_block4 = Middle_stream_block(128, 64, 32, frag=True)
        self.ups1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.ups2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.ups3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, fore_e, back_e, fore_d, back_d, fore_m, back_m, input_pos1, input_pos2, input_pos3, input_pos4):
        out_pos1, out_back1, out_shape1 = self.ms_block1(fore_e[0], back_e[0], fore_d[0], back_d[0], fore_m, back_m,
                                                         input_pos1)
        out_pos2, out_back2, out_shape2 = self.ms_block2(fore_e[1], back_e[1], fore_d[1], back_d[1], out_pos1, out_back1,
                                                         input_pos2)
        out_pos3, out_back3, out_shape3 = self.ms_block3(fore_e[2], back_e[2], fore_d[2], back_d[2], out_pos2, out_back2,
                                                         input_pos3)
        out_middle, out_back4, out_shape4 = self.ms_block4(fore_e[3], back_e[3], fore_d[3], back_d[3], out_pos3, out_back3,
                                                         input_pos4)
        out_shape1 = self.ups1(out_shape1)
        out_shape2 = self.ups2(out_shape2)
        out_shape3 = self.ups3(out_shape3)
        out_shape = torch.cat([out_shape1, out_shape2, out_shape3, out_shape4], dim=1)
        return out_middle, out_shape

class SMU_Net(nn.Module):
    def __init__(self):
        super(SMU_Net, self).__init__()
        self.encode = encode()
        self.decode = decode()
        self.mid = Middle_stream()
        self.conv_f = nn.Conv2d(64, 1, (1, 1))
        self.conv_s = nn.Conv2d(64, 1, (1, 1))
        self.conv_m = nn.Conv2d(64, 1, (1, 1))
        self.sig = nn.Sigmoid()
        self.initialize_weights()

    def forward(self,input_ori, input_fore, input_back, input_pos1, input_pos2, input_pos3, input_pos4):
        fore_input = torch.cat([input_ori, input_fore], dim=1)
        back_input = torch.cat([input_ori, input_back], dim=1)
        fore_e, fore_m = self.encode(fore_input)
        back_e, back_m = self.encode(back_input)
        fore_d = self.decode(fore_m, fore_e[0], fore_e[1], fore_e[2], fore_e[3])
        back_d = self.decode(back_m, back_e[0], back_e[1], back_e[2], back_e[3])
        out_middle, out_shape = self.mid(fore_e, back_e, fore_d, back_d, fore_m, back_m, input_pos1, input_pos2, input_pos3, input_pos4)
        out_fore = self.sig(self.conv_f(fore_d[3]))
        out_shape = self.sig(self.conv_s(out_shape))
        out_middle = self.sig(self.conv_m(out_middle))
        return out_fore, out_shape, out_middle

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)










