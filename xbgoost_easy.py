import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the datasets
df = pd.read_csv('ais_train.csv', sep='|')
test_df = pd.read_csv('ais_test.csv', sep=',')

# Sort by vesselId and time to ensure proper time-series ordering for each vessel
df = df.sort_values(by=['vesselId', 'time'])

# Generate lagged features for the training data
df['latitude_t-1'] = df.groupby('vesselId')['latitude'].shift(1)
df['longitude_t-1'] = df.groupby('vesselId')['longitude'].shift(1)
df['sog_t-1'] = df.groupby('vesselId')['sog'].shift(1)
df['cog_t-1'] = df.groupby('vesselId')['cog'].shift(1)
df['heading_t-1'] = df.groupby('vesselId')['heading'].shift(1)

# Drop rows with NaN values
df = df.dropna()

# Define X (features) and y (targets) for training
X = df[['latitude_t-1', 'longitude_t-1', 'sog_t-1', 'cog_t-1', 'heading_t-1']]
y_latitude = df['latitude']
y_longitude = df['longitude']

# Train-test split for validation
X_train, X_val, y_lat_train, y_lat_val = train_test_split(X, y_latitude, test_size=0.2, random_state=42)
X_train, X_val, y_lon_train, y_lon_val = train_test_split(X, y_longitude, test_size=0.2, random_state=42)

# Train XGBoost model for latitude prediction
model_lat = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
model_lat.fit(X_train, y_lat_train)

# Train XGBoost model for longitude prediction
model_lon = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
model_lon.fit(X_train, y_lon_train)

# ---- Test Set Processing ---- #

# Sort test data by vesselId and time to align it
test_df = test_df.sort_values(by=['vesselId', 'time'])

# Merge the last known position (from training data) with the test data based on vesselId
# We'll use this to create the lag features for the test set.
latest_known_data = df.groupby('vesselId').last().reset_index()

# Merge the test set with the last known positions from the training data
test_df = test_df.merge(latest_known_data[['vesselId', 'latitude', 'longitude', 'sog', 'cog', 'heading']], on='vesselId', how='left')

# Now rename these columns to represent the t-1 features
test_df['latitude_t-1'] = test_df['latitude']
test_df['longitude_t-1'] = test_df['longitude']
test_df['sog_t-1'] = test_df['sog']
test_df['cog_t-1'] = test_df['cog']
test_df['heading_t-1'] = test_df['heading']

# Drop the extra columns after generating the lagged features
test_df = test_df.drop(columns=['latitude', 'longitude', 'sog', 'cog', 'heading'])

# Handle missing values in test set (if any remain)
test_df.fillna(method='ffill', inplace=True)

# Features for prediction
X_test = test_df[['latitude_t-1', 'longitude_t-1', 'sog_t-1', 'cog_t-1', 'heading_t-1']]

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

print(df.columns)