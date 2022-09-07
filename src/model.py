import torch
import torch.nn as nn
import numpy as np


class TabularModel(nn.Module):

    def __init__(self):
        super().__init__()