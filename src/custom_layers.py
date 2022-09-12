import torch.nn as nn


class outLayer1d(nn.Module):

    def __init__(self, in_features, output_size, activation=None):
        super().__init__()
        self.sl1 = nn.Linear(in_features, output_size)
        self.sl2 = activation if activation else lambda x: x

    def forward(self, x):
        x = self.sl1(x)
        x = self.sl2(x)
        return x
