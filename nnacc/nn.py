from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        if in_size == out_size:
            self.skip_layer = nn.Identity()
        else:
            self.skip_layer = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x):
        h = self.layer1(F.relu(x))
        y = self.layer2(F.relu(h)) * 0.1 + self.skip_layer(x)

        return y
