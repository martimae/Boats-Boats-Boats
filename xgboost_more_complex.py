import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the datasets
df = pd.read_csv('ais_train.csv', sep='|')
test_df = pd.read_csv('ais_test.csv', sep=',')

# Sort by vesselId and time to ensure proper time-series ordering for each vessel
df = df.sort_values(by=['vesselId', 'time'])

# Generate lagged features for the training data
lag_features_latitude = [1, 2]  # Generate lagged features for latitude (t-1, t-2)
lag_features_longitude = [1]     # Generate lagged feature for longitude (t-1)

# Create lagged features for latitude
for lag in lag_features_latitude:
    df[f'latitude_t-{lag}'] = df.groupby('vesselId')['latitude'].shift(lag)
    df[f'sog_t-{lag}'] = df.groupby('vesselId')['sog'].shift(lag)
    df[f'cog_t-{lag}'] = df.groupby('vesselId')['cog'].shift(lag)
    df[f'heading_t-{lag}'] = df.groupby('vesselId')['heading'].shift(lag)

# Create lagged features for longitude
for lag in lag_features_longitude:
    df[f'longitude_t-{lag}'] = df.groupby('vesselId')['longitude'].shift(lag)

# Create rolling averages (3-point rolling window)
df['sog_rolling_avg'] = df.groupby('vesselId')['sog'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df['cog_rolling_avg'] = df.groupby('vesselId')['cog'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Extract time-based features (assuming 'time' is in a proper datetime format)
df['time'] = pd.to_datetime(df['time'])  # Convert time column to datetime
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek

# Drop rows with NaN values
df = df.dropna()

# Define X (features) and y (targets) for training
features = [f'latitude_t-{lag}' for lag in lag_features_latitude] + \
           [f'longitude_t-{lag}' for lag in lag_features_longitude] + \
           [f'sog_t-{lag}' for lag in lag_features_latitude] + \
           [f'cog_t-{lag}' for lag in lag_features_latitude] + \
           [f'heading_t-{lag}' for lag in lag_features_latitude] + \
           ['sog_rolling_avg', 'cog_rolling_avg', 'hour', 'day_of_week']

X = df[features]
y_latitude = df['latitude']
y_longitude = df['longitude']

# Train-test split for validation
X_train, X_val, y_lat_train, y_lat_val = train_test_split(X, y_latitude, test_size=0.2, random_state=42)
X_train_lon, X_val_lon, y_lon_train, y_lon_val = train_test_split(X, y_longitude, test_size=0.2, random_state=42)

# Train XGBoost model for latitude prediction
model_lat = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=145, max_depth=5)
model_lat.fit(X_train, y_lat_train)

# Predict latitude and calculate MSE for latitude
lat_pred = model_lat.predict(X_val)
lat_mse = mean_squared_error(y_lat_val, lat_pred)
print(f"Mean Squared Error for Latitude: {lat_mse}")

# Train XGBoost model for longitude prediction
model_lon = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=145, max_depth=5)
model_lon.fit(X_train_lon, y_lon_train)

# Predict longitude and calculate MSE for longitude
lon_pred = model_lon.predict(X_val_lon)
lon_mse = mean_squared_error(y_lon_val, lon_pred)
print(f"Mean Squared Error for Longitude: {lon_mse}")

# ---- Test Set Processing ---- #

# Sort test data by vesselId and time to align it
test_df = test_df.sort_values(by=['vesselId', 'time'])

# Merge the last known position (from training data) with the test data based on vesselId
latest_known_data = df.groupby('vesselId').last().reset_index()

# Merge the test set with the last known positions from the training data
test_df = test_df.merge(latest_known_data[['vesselId', 'latitude', 'longitude', 'sog', 'cog', 'heading']], on='vesselId', how='left')

# Create lagged features for latitude in the test set
for lag in lag_features_latitude:
    test_df[f'latitude_t-{lag}'] = test_df['latitude']
    test_df[f'sog_t-{lag}'] = test_df['sog']
    test_df[f'cog_t-{lag}'] = test_df['cog']
    test_df[f'heading_t-{lag}'] = test_df['heading']

# Create lagged features for longitude in the test set
for lag in lag_features_longitude:
    test_df[f'longitude_t-{lag}'] = test_df['longitude']

# Create rolling averages for the test set
test_df['sog_rolling_avg'] = test_df.groupby('vesselId')['sog'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
test_df['cog_rolling_avg'] = test_df.groupby('vesselId')['cog'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Extract time-based features from test set
test_df['time'] = pd.to_datetime(test_df['time'])  # Convert time column to datetime
test_df['hour'] = test_df['time'].dt.hour
test_df['day_of_week'] = test_df['time'].dt.dayofweek

# Drop the extra columns after generating the lagged features
test_df = test_df.drop(columns=['latitude', 'longitude', 'sog', 'cog', 'heading'])

# Handle missing values in test set (if any remain)
test_df.fillna(method='ffill', inplace=True)

# Features for prediction
X_test = test_df[features]  # Use the same features as in training

# Predict longitude and latitude using the trained models
pred_lat_test = model_lat.predict(X_test)
pred_lon_test = model_lon.predict(X_test)

# Add predictions to the test dataframe
test_df['latitude_predicted'] = pred_lat_test
test_df['longitude_predicted'] = pred_lon_test

# Format the final output
output_df = test_df[['ID', 'longitude_predicted', 'latitude_predicted']]

# Save the results to a CSV file
output_df.to_csv('predictions.csv', index=False)


