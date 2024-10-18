import pandas as pd
import xgboost as xgb

# Load the datasets
df = pd.read_csv('train_split_time.csv')
test_df1 = pd.read_csv('test_set_no_lat_lon.csv')

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

# Train XGBoost models
model_lat = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=5, max_depth=5)
model_lon = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=5, max_depth=5)

# Train models for latitude and longitude
model_lat.fit(X, y_latitude)
model_lon.fit(X, y_longitude)

# ---- Test Set Processing ---- #

# Sort test data by vesselId and time to align it
test_df1 = test_df1.sort_values(by=['vesselId', 'time'])

test_df = test_df1.head(112)

# Merge the last known position (from training data) with the test data based on vesselId
latest_known_data = df.groupby('vesselId').last().reset_index()
test_df = test_df.merge(latest_known_data[['vesselId', 'latitude', 'longitude', 'sog', 'cog', 'heading']], on='vesselId', how='left')

# Rename these columns to represent the t-1 features
test_df['latitude_t-1'] = test_df['latitude']
test_df['longitude_t-1'] = test_df['longitude']
test_df['sog_t-1'] = test_df['sog']
test_df['cog_t-1'] = test_df['cog']
test_df['heading_t-1'] = test_df['heading']

# Drop the extra columns after generating the lagged features
test_df = test_df.drop(columns=['latitude', 'longitude', 'sog', 'cog', 'heading'])

# Features for prediction
X_test = test_df[['latitude_t-1', 'longitude_t-1', 'sog_t-1', 'cog_t-1', 'heading_t-1']]

# ---- Iterative Predictions ---- #

# Initialize lists to store predictions
pred_lat_list = []
pred_lon_list = []

df = pd.DataFrame(columns=['ID','time','vesselId','scaling_factor','latitude_t-1','longitude_t-1','sog_t-1','cog_t-1','heading_t-1','latitude_predicted','longitude_predicted'])

# Iterate over the test set row by row and predict the next timestep
for i in range(len(X_test)):
    # Select the current row for prediction
    current_row = X_test.iloc[i].values.reshape(1, -1)
    
    # Predict latitude and longitude for the current timestep
    pred_lat = model_lat.predict(current_row)
    pred_lon = model_lon.predict(current_row)
    
    # Store predictions
    pred_lat_list.append(pred_lat[0])
    pred_lon_list.append(pred_lon[0])
    
    # Print debug information
    print(f"Iteration {i}: Current Row Values - {X_test.iloc[i].to_dict()}")
    print(f"Iteration {i}: Predicted Latitude - {pred_lat[0]}, Predicted Longitude - {pred_lon[0]}")
    
    # Update t-1 features for the next row
    if i < len(X_test) - 1:  # Avoid updating beyond the last row
        X_test.iloc[i + 1, X_test.columns.get_loc('latitude_t-1')] = pred_lat[0]
        X_test.iloc[i + 1, X_test.columns.get_loc('longitude_t-1')] = pred_lon[0]

# Add the predictions to the test dataframe
test_df['latitude_predicted'] = pred_lat_list
test_df['longitude_predicted'] = pred_lon_list

# Format the final output
output_df = test_df[['ID', 'longitude_predicted', 'latitude_predicted']]
out2 = test_df

# Save the results to a CSV file
output_df.to_csv('predictions.csv', index=False)
out2.to_csv('pred_all.csv', index=False)
