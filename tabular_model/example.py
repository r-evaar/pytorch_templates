import pandas as pd
from src.preprocess_nyc import process_nyc_dataset
from src.preprocess import pd_to_torch

nyc = pd.read_csv('./data/NYCTaxiFares.csv')
process_nyc_dataset(nyc)

categorical_features = ['Hour','AMorPM','Weekday']
continuous_features = ['pickup_longitude','pickup_latitude', 'dropoff_longitude',
                       'dropoff_latitude', 'passenger_count', 'dist_km']
y_feature = 'fare_amount'

cat, cont, y = pd_to_torch(nyc, categorical_features, continuous_features, y_feature, gpu=True)

