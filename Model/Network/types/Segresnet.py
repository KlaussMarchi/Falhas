import torch.nn.functional as F
import torch
import torch.optim as optim
from monai.networks.nets import SegResNet


def Segresnet(channels, classes, num_filters, dropout):
    return SegResNet(spatial_dims=3, in_channels=channels, out_channels=classes, init_filters=num_filters, dropout_prob=dropout)
