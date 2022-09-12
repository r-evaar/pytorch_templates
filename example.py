import pandas as pd
from src.custom_preprocess import process_nyc_dataset
from src.preprocess import pd_to_torch
from src.models import TabularModel

# Data Loading and Pre-Processing
# nyc = pd.read_csv('./data/NYCTaxiFares.csv')
# process_nyc_dataset(nyc)
nyc = pd.read_csv('./tests/NYCTaxiFaresPreProcessed.csv')

# Identifying Features (cat & cont) in the tabular dataframe
categorical_features = ['Hour', 'AMorPM', 'Weekday']
continuous_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                       'dropoff_latitude', 'passenger_count', 'dist_km']
y_feature = 'fare_amount'

classification = False

# Loading data as tensors. Tensors are created on a CUDA device if one is available
cat, cont, y = pd_to_torch(nyc, categorical_features, continuous_features, y_feature,
                           categorical_target=classification, gpu=True)

model = TabularModel(layers=[50, 100, 50], classification=classification)
model.fit(cat, cont, y, ratios=[0.8, 0.1, 0.1])

d = model.train_loader._get_iterator().next()
x = model.forward(d[0], d[1])
x

