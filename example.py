import pandas as pd
from src.custom_preprocess import process_nyc_dataset
from src.preprocess import pd_to_torch
from src.models import TabularModel
import torch
import matplotlib.pyplot as plt

LOAD_PREPROCESSED = True  # Custom preprocessing takes time. Set it to 'True' when debugging

# Data Loading with/without custom Pre-Processing
if LOAD_PREPROCESSED:
    nyc = pd.read_csv('./tests/NYCTaxiFaresPreProcessed.csv')
else:
    nyc = pd.read_csv('./data/NYCTaxiFares.csv')
    process_nyc_dataset(nyc)

# Identifying Features (cat & cont) in the tabular dataframe
categorical_features = ['Hour', 'AMorPM', 'Weekday']
continuous_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                       'dropoff_latitude', 'passenger_count', 'dist_km']
y_feature = 'fare_amount'

# Problem type [classification/regression]
classification = False

# Loading data as tensors. Tensors are created on a CUDA device if one is available
cat, cont, y = pd_to_torch(nyc, categorical_features, continuous_features, y_feature,
                           categorical_target=classification, gpu=True)

# Model initialization
model = TabularModel(layers=[50, 100, 50], classification=classification)

# Fitting NYC dataset to the model
model.configs.batch_size = 4096
model.fit(cat, cont, y, ratios=[0.8, 0.1, 0.1])  # Create dataloaders internally from the NYC dataset tensors

# RMSE cost function declaration
cost_func = lambda *args: torch.sqrt(torch.nn.functional.mse_loss(*args))

# Train and plot on fit data
epochs = 20
loss_record = model.train_model(cost_func, epochs=epochs)
plt.plot(range(epochs), loss_record); plt.xlabel('Epochs'); plt.ylabel('loss'); plt.show()

# Save trained model parameters
torch.save(model.state_dict(), './model/TaxiFarePredictor.pt')



