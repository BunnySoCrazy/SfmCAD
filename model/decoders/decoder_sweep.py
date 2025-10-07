import torch
import torch.nn.functional as F
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, ef_dim=32, num_primitives=4, segment=8, drop_rate=0.2):
        super(Decoder, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = ef_dim
        self.segment = segment
        self.drop_rate = drop_rate
        self.n_control_p = 4
        self.n_coord = 3
        self.linear_1 = nn.Linear(self.feature_dim*8, self.feature_dim*8, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        self.dropout = nn.Dropout(self.drop_rate)

        self.primitive_linear = nn.Linear(self.feature_dim*8,
                                          int(num_primitives*(self.n_control_p*self.n_coord+2+2)),
                                          bias=True)
        nn.init.xavier_uniform_(self.primitive_linear.weight)
        nn.init.constant_(self.primitive_linear.bias, 0)

    def forward(self, feature):
        l1 = self.linear_1(feature)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)
        l1 = self.dropout(l1)

        shapes = self.primitive_linear(l1+feature)

        return shapes
