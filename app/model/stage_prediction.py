import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the dataset
file_path = "/content/drive/My Drive/stage_predict_dataset.csv"
data = pd.read_csv(file_path)

# Clean and preprocess the data
data['DATE'] = data['DATE'].fillna(method='ffill')
data['DATE'] = pd.to_datetime(data['DATE'], format='%d/%m/%Y', errors='coerce')
data = data.dropna(subset=['DATE'])
data.fillna(method='ffill', inplace=True)

# Select features and target
features = data[['Raw_Water_Turbidity', 'Raw_Water_PH', 'Raw_Water_Conductivity']]
target = data[['Treated_Water_Turbidity', 'Treated_Water_PH', 'Treated_Water_Conductivity']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler mean and scale for use in Spring Boot
scaler_mean = scaler.mean_.reshape(1, -1)
scaler_scale = scaler.scale_.reshape(1, -1)
joblib.dump(scaler_mean, '/content/drive/My Drive/scaler_mean.pkl')
joblib.dump(scaler_scale, '/content/drive/My Drive/scaler_scale.pkl')

# Build and train the regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Save the model for Spring Boot integration
joblib.dump(model, '/content/drive/My Drive/stage_prediction_model.pkl')
# Save the scaler for inverse transformation in Spring Boot
joblib.dump(scaler, '/content/drive/My Drive/scaler.pkl')

# Function to predict treated water values
def predict_treated_water(raw_turbidity, raw_ph, raw_conductivity):
    input_data = pd.DataFrame([[raw_turbidity, raw_ph, raw_conductivity]], columns=['Raw_Water_Turbidity', 'Raw_Water_PH', 'Raw_Water_Conductivity'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Example usage
raw_turbidity = 6
raw_ph = 7
raw_conductivity = 3
treated_water_values = predict_treated_water(raw_turbidity, raw_ph, raw_conductivity)
print(f"Predicted Treated Water Values: Turbidity={treated_water_values[0]}, pH={treated_water_values[1]}, Conductivity={treated_water_values[2]}")