import torch

import torch.nn as nn
import warnings
from dataclasses import dataclass
from src.custom_layers import outLayer1d
from torch.utils.data import TensorDataset, random_split, DataLoader
from utils.data import ratios_to_integers


class TabularModel(nn.Module):

    def configure(self, activation=nn.ReLU, batch_norm=True, dropout=0.05):
        @dataclass
        class configs:
            cfg_activation = activation
            cfg_batch_norm = nn.BatchNorm1d if batch_norm else lambda x: x
            cfg_dropout = nn.Dropout(dropout)
            cfg_batch_size = 512

        self.configs = configs

    def __init__(self, layers, out_size=1, classification=False):
        super().__init__()
        self.cat = False    # Model will process categorical input
        self.cont = False   # Model will process continuous input
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
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

        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        activation = nn.Softmax() if classification else None
        self.out_layer = outLayer1d(in_features, out_size, activation=activation)

    def fit(self, cat_data=None, cont_data=None, y=None, ratios=None, reset_weights=True):
        # 'ratios' default value
        if ratios is None:
            ratios = [0.7, 0.2, 0.1]

        # Make sure that 'ratios' is a list
        if type(ratios) is not list:
            ratios = [ratios]

        assert len(ratios) in [1, 2, 3], "You can split into" \
                                         "\n[train, val, test]" \
                                         "\n[train, test]" \
                                         "or set 'ratios' to 1 for no splitting"
        assert sum(ratios) == 1, "Splitting ratios must add up to 1"
        assert cat_data or cont_data and y, \
            "Training set must have at least one input feature type and must have a target"
        self.is_fit = True  # After assertions

        # in_features at layer 0 are set to 0 in '__init__'
        # The first layer will change based on the training data
        # NOTE: Using 'fit()' will reset the model weights by default

        n_embeddings = 0
        n_cont_features = 0

        data = []
        if cat_data is not None:
            self.cat = True
            cat_tensor = cat_data[0]
            embedding_sizes = cat_data[1]
            data.append(cat_tensor)  # Tensor at index 0
            n_embeddings = sum([nf for _, nf in embedding_sizes])

            # Create the embedding layers for categorical data.
            # One-hot-encodes the input data before the first layer
            self.embed_layers = nn.ModuleList([
                nn.Embedding(ni, nf) for ni, nf in embedding_sizes
            ])

        if cont_data is not None:
            self.cont = True
            data.append(cont_data)
            n_cont_features = cont_data.shape[1]
            self.cont_batch_norm = self.configs.cfg_batch_norm(n_cont_features)

        data.append(y)
        device = y.device.type

        total_in_features = n_embeddings + n_cont_features

        self.hidden_layers[0] = nn.Linear(
            total_in_features, self.hidden_layers[0].out_features
        )  # Resetting number of input features

        # Resetting weights because the neural synapse has been altered
        # Retraining is required for a functioning network
        if reset_weights:  # True by default
            # Apply to all trainable model layers
            self.apply(lambda layer: layer.reset_parameters() \
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d) \
                else None)

        dataset = TensorDataset(*data)
        m = len(dataset)

        set_sizes = ratios_to_integers(m, ratios)

        self.splits = 1 if ratios == [1] else 2 if len(ratios) == 2 else 3
        subsets = random_split(dataset, set_sizes)

        self.train_loader = DataLoader(subsets[0], self.configs.cfg_batch_size)

        # Development Note: Requires checking if maximum batch_size can be handled by memory
        if self.splits == 2:
            self.test_loader = DataLoader(subsets[1], set_sizes[1])
        if self.splits == 3:
            self.val_loader = DataLoader(subsets[1], set_sizes[1])
            self.test_loader = DataLoader(subsets[2], set_sizes[2])

        if device == 'cuda':
            self.cuda()

    def forward(self, x_cat=None, x_cont=None):
        assert x_cat is not None or x_cont is not None, "No data to propagate"

        ### Preparing categorical and continuous input for the first layer
        x = []

        if self.cat:
            embeddings = []
            embeddings.extend([
                layer(x_cat[:, feature]) for feature, layer in enumerate(self.embed_layers)
            ])
            x_cat = torch.cat(embeddings, dim=1)
            x_cat = self.configs.cfg_dropout(x_cat)
            x.append(x_cat)

        if self.cont:
            x_cont = self.cont_batch_norm(x_cont)
            x.append(x_cont)

        x = torch.cat(x, dim=1)  # Convert x from a list of tensors into a concatenated tensor

        ### Forward Propagation
        x = self.hidden_layers(x)
        x = self.out_layer(x)

        return x


