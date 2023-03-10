import torch
import random
import itertools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Parameter

class rsaBlock(nn.Module):

    def __init__(self, in_channels):

        super(rsaBlock, self).__init__()

        self.theta_conv = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.phi_conv   = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.g_conv     = nn.Conv3d(in_channels, in_channels, 1, 1, 0)

        #coronal_sa  = saCoronalBlock()
        #sagittal_sa = saSagittalBlock()
        self.axial_sa    = saAxialBlock()

        #coronal_beta  = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        #sagittal_beta = nn.Parameter(torch.tensor(0.0, requires_grad = True))
        axial_beta    = nn.Parameter(torch.tensor(0.0, requires_grad = True))

        #self.saBlocks = nn.ModuleList([coronal_sa, sagittal_sa, axial_sa])
        #self.betas = nn.ParameterList([coronal_beta, sagittal_beta, axial_beta])
        #self.betas = nn.ParameterList(axial_beta)

    def forward(self, x):

        theta_x = self.theta_conv(x)
        phi_x = self.phi_conv(x)
        g_x = self.g_conv(x)

        x = x + self.axial_sa(theta_x, phi_x, g_x) * axial_beta

        return x

class saAxialBlock(nn.Module):

    def __init__(self):

        super(saAxialBlock, self).__init__()

    def forward(self, x_, x_t, g_x):
        print('theta', x_.shape)
        print('phi', x_t.shape)
        print('g', g_x.shape)

        batch_size = x_.size()[0] # original size: n * c * d * h * w
        depth_size = x_.size()[1]

        #x_ = x_.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_ = x_.contiguous().view(batch_size, depth_size, -1)
        # n * c * hw

        #g_x = g_x.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, depth_size, -1)
        # n * c * hw

        #x_t = x_t.permute(0, 2, 1, 3, 4)
        # n * d * c * h * w
        x_t = x_t.contiguous().view(batch_size, depth_size, -1)
        # n * c * hw
        x_t = x_t.permute(0, 2, 1)
        # n * hw * c

        attention = torch.matmul(x_, x_t)
        # n * c * c
        attention = F.softmax(attention, dim = -1)
        # n * c * c
        del x_, x_t

        # n * c * c X n * c * hw
        out = torch.matmul(attention, g_x)
        # n * d * chw
        del attention, g_x
        out = out.view(batch_size, depth_size, *ori_size[2:])
        # n * c * h * w
        #out = out.permute(0, 2, 1, 3, 4)
        # n * c * d * h * w

        return out


class saCoronalBlock(nn.Module):

    def __init__(self):

        super(saCoronalBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0]
        coronal_size = x_.size()[3]

        # n * c * d * h * w
        x_ = x_.permute(0, 3, 2, 1, 4)
        x_ = x_.contiguous().view(batch_size, coronal_size, -1)

        g_x = g_x.permute(0, 3, 2, 1, 4)
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, coronal_size, -1)

        x_t = x_t.permute(0, 3, 2, 1, 4)
        x_t = x_t.contiguous().view(batch_size, coronal_size, -1)
        x_t = x_t.permute(0, 2, 1)

        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        del x_, x_t

        out = torch.matmul(attention, g_x)
        del attention, g_x
        out = out.view(batch_size, coronal_size, *ori_size[2:])
        out = out.permute(0, 3, 2, 1, 4)

        return out

class saSagittalBlock(nn.Module):

    def __init__(self):

        super(saSagittalBlock, self).__init__()

    def forward(self, x_, x_t, g_x):

        batch_size = x_.size()[0]
        sagittal_size = x_.size()[4]

        # n * c * d * h * w
        x_ = x_.permute(0, 4, 2, 3, 1)
        x_ = x_.contiguous().view(batch_size, sagittal_size, -1)

        g_x = g_x.permute(0, 4, 2, 3, 1)
        ori_size = g_x.size()
        g_x = g_x.contiguous().view(batch_size, sagittal_size, -1)

        x_t = x_t.permute(0, 4, 2, 3, 1)
        x_t = x_t.contiguous().view(batch_size, sagittal_size, -1)
        x_t = x_t.permute(0, 2, 1)

        attention = torch.matmul(x_, x_t)
        attention = F.softmax(attention, dim = -1)
        del x_, x_t

        out = torch.matmul(attention, g_x)
        del attention, g_x
        out = out.view(batch_size, sagittal_size, *ori_size[2:])
        out = out.permute(0, 4, 2, 3, 1)

        return out
