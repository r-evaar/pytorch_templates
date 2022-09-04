import numpy as np
import pandas as pd
from tabular_model.utils.measurements import haversine


def process_nyc_dataset(df):
    print('Processing NYC dataset .. ', end='')

    earth_radius = 6.3781e3
    haversine_on_earth = haversine(earth_radius)

    # New continuous feature (dist_km)
    df['dist_km'] = haversine_on_earth(
        (df['pickup_longitude'], df['pickup_latitude']),
        (df['dropoff_longitude'], df['dropoff_latitude'])
    )

    # New Features (Hour, AMorPM, Weekday)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])  # Convert to datetime objects
    df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)  # Compensate for time difference
    #
    #
    df['Hour'] = df['EDTdate'].dt.hour  # pd.Series.dt method, returns hour as an integer
    df['AMorPM'] = np.where(df['Hour'] > 12, 'PM', 'AM')
    df['Weekday'] = df['EDTdate'].dt.strftime("%a")  # pd.Series.dt method, returns week day as a string

    print('done')


# Testing
if __name__ == '__main__':
    df = pd.read_csv('../Data/NYCTaxiFares.csv')
    process_nyc_dataset(df)
    pd.set_option('display.max_columns', None)
    print(df.head())
