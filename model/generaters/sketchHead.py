import torch
import torch.nn as nn
import numpy as np

class SketchHead(nn.Module):
    def __init__(self, d_in, dims):
        super().__init__()
        dims = [d_in] + dims + [1]
        self.num_layers = len(dims)
        for layer in range(0, self.num_layers - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if layer == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                torch.nn.init.constant_(lin.bias, -1)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, "lin" + str(layer), lin)
            self.activation = nn.Softplus(beta=100)

    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
            else:
                x = x.clamp(-1,1)
        return x
