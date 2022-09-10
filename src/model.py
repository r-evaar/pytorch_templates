import torch
import torch.nn as nn
import numpy as np
import warnings
from dataclasses import dataclass
from utils.custom_layers import outLayer1d


class TabularModel(nn.Module):

    def configure(self, activation=nn.ReLU, batch_norm=True, dropout=0.05):
        @dataclass
        class configs:
            cfg_activation = activation
            cfg_batch_norm = nn.BatchNorm1d if batch_norm else lambda x: x
            cfg_dropout = nn.Dropout(dropout)

        self.configs = configs

    def __init__(self, layers, out_size=1, classification=False):
        super().__init__()
        self.is_fit = False
        self.configure()

        self.hidden_layers = []

        warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")
        in_features = 0

        for out_features in layers:
            self.hidden_layers.append(nn.Linear(in_features, out_features))
            self.hidden_layers.append(self.configs.cfg_activation(inplace=True))
            self.hidden_layers.append(self.configs.cfg_batch_norm(out_features))
            self.hidden_layers.append(self.configs.cfg_dropout)
            in_features = out_features

        activation = nn.Softmax() if classification else None
        self.out_layer = outLayer1d(in_features, out_size, activation=activation)

