import numpy as np
import pandas as pd
import torch
from utils.resources import device_name


def pd_to_torch(df, cat_features=None, cont_features=None, y_feature=None, categorical_target=False, gpu=False):

    # x_features are lists containing the column names for categorical (cat), continuous (cont), and target data series
    # in the dataframe (df). x_features can be a string if a single column is selected.
    for features in [cat_features, cont_features, y_feature]:
        if features is not None and type(features) != list and type(features) != str:
            raise ValueError(f'{features} cannot be used to access dataframe column(s)')

    has_cat = cat_features is not None
    has_cont = cont_features is not None
    has_y = y_feature is not None

    out = []
    if has_cat:
        df[cat_features] = df[cat_features].astype('category')  # Enable access to pd.Series.cat methods for cat_features
        X_cat_numpy = np.stack([df[feature].cat.codes.values for feature in cat_features], axis=1)
        X_cat = torch.tensor(X_cat_numpy, dtype=torch.int64, device=device_name(gpu))

        # Preparing pairs of io sizes for categorical input
        # required for one-hot input encoding using torch.nn.embedding
        embedding_io = []
        for feature in cat_features:
            enumerated_categories = df[feature].cat.categories
            size = len(enumerated_categories)

            embedding_io.append(
                (size, min(50, (size + 1) // 2))
            )

        cat_data = (X_cat, embedding_io)
        out.append(cat_data)

    if has_cont:
        X_cont_numpy = np.stack([df[feature].values for feature in cont_features], axis=1)
        cont_data = torch.tensor(X_cont_numpy, dtype=torch.float32, device=device_name(gpu))
        out.append(cont_data)

    if has_y:
        y_numpy = df[y_feature].values

        target_type = torch.int64 if categorical_target else torch.float32
        target = torch.tensor(y_numpy, dtype=target_type, device=device_name(gpu))

        if categorical_target:
            target = target.view(-1)

        out.append(target)

    return out if out else None
