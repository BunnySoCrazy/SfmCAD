import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, ef_dim=32):
        super(Encoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*8, 4, stride=1, padding=0, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, inputs):
        d = inputs
        for i in range(1, 6):
            d = getattr(self, f"conv_{i}")(d)
            d = F.leaky_relu(d, negative_slope=0.01)

        d = d.view(-1, self.ef_dim*8)
        d = F.leaky_relu(d, negative_slope=0.01)
        return d


class Encoder_(nn.Module):
    def __init__(self, ef_dim=32):
        super(Encoder_, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*8, 4, stride=2, padding=2, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, inputs):
        d = inputs
        residuals = []
        for i in range(1, 6):
            d = getattr(self, f"conv_{i}")(d)
            d = F.leaky_relu(d, negative_slope=0.01)
            residuals.append(d)

        d = d + residuals[-2]
        d = d.view(-1, self.ef_dim*8)
        d = F.leaky_relu(d, negative_slope=0.01)
        return d
