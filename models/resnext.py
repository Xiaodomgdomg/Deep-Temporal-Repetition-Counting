import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, kernels=3):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=kernels,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, opt):
        # default
        shortcut_type='B'
        cardinality=32
        num_classes=400

        # user paras
        num_classes=opt.n_classes
        shortcut_type=opt.resnet_shortcut
        cardinality=opt.resnext_cardinality
        sample_size=opt.sample_size
        sample_duration=opt.basic_duration
        self.learning_policy=opt.learning_policy
        self.num_classes = opt.n_classes
        self.inplanes = 64
        super(ResNeXt, self).__init__()

        down_stride_1 = (1, 2, 2) 
        down_stride_2 = (2, 2, 2) 

        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=down_stride_1,
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        base_c = 128

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=down_stride_2, padding=1)
        self.layer1 = self._make_layer(
            block, base_c, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(
            block, base_c*2, layers[1], shortcut_type, cardinality, stride=down_stride_2)
        self.layer3 = self._make_layer(
            block, base_c*4, layers[2], shortcut_type, cardinality, stride=down_stride_2)
        self.layer4 = self._make_layer(
            block, base_c*8, layers[3], shortcut_type, cardinality, stride=down_stride_2)

        last_duration = int(1)
        last_size = int(math.ceil(sample_size / 32.0))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        self.t_all =  int(sample_duration / 16.0)
        self.dims = int(base_c*8 * block.expansion)

        self.fc_emd = self.dims * self.t_all / 2

        if self.learning_policy == '2stream':
            self.fc_cls_1 = nn.Linear(self.fc_emd, 2*num_classes).cuda()
            self.fc_box_1 = nn.Linear(self.fc_emd, num_classes).cuda()
            self.fc_cls_2 = nn.Linear(self.fc_emd, 2*num_classes).cuda()
            self.fc_box_2 = nn.Linear(self.fc_emd, num_classes).cuda()
        
        self.fc = []
        self.others = []
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1,
                    kernels=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        print_flag = False
        if print_flag:
            print('x1 ', x.size())
        x = self.conv1(x)
        if print_flag:
            print('x2 ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.avgpool(x5)

        new_x_1 = x[:,:,0:self.t_all/2,:,:]
        new_x_2 = x[:,:,self.t_all/2:self.t_all,:,:]

        new_x_1 = new_x_1.reshape(-1, self.fc_emd)
        new_x_2 = new_x_2.reshape(-1, self.fc_emd)
       
        if self.learning_policy == '2stream':
            pred_cls_1 = self.fc_cls_1(new_x_1)
            pred_cls_1 = pred_cls_1.reshape(-1, 2, self.num_classes)
            pred_box_1 = self.fc_box_1(new_x_1)

            pred_cls_2 = self.fc_cls_2(new_x_2)
            pred_cls_2 = pred_cls_2.reshape(-1, 2, self.num_classes)
            pred_box_2 = self.fc_box_2(new_x_2)
            return pred_cls_1, pred_box_1, pred_cls_2, pred_box_2




def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
