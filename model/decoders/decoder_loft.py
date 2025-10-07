import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import add_latent


class Decoder(nn.Module):
    def __init__(self, ef_dim=32, num_primitives=4, segment=8):
        super(Decoder, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = ef_dim
        self.segment = segment
        self.n_control_p = 4
        self.n_coord = 3
        self.linear_1 = nn.Linear(self.feature_dim*8, self.feature_dim*16, bias=True)
        self.linear_2 = nn.Linear(self.feature_dim*16, self.feature_dim*32, bias=True)
        self.linear_3 = nn.Linear(self.feature_dim*32, self.feature_dim*64, bias=True)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)

        self.primitive_linear = nn.Linear(self.feature_dim*64,
                                          int(num_primitives*(256+8)),
                                          bias=True)
        nn.init.xavier_uniform_(self.primitive_linear.weight)
        nn.init.constant_(self.primitive_linear.bias, 0)

    def forward(self, feature):
        B = feature.shape[0]
        l1 = self.linear_1(feature)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        shapes = self.primitive_linear(l3)
        return shapes.reshape(B, self.num_primitives, 256+8)
