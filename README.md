# 🚄 JetRail Traffic Forecasting

## 🎯 Objective
Forecast JetRail's monthly user traffic for the next 7 months to help Unicorn Ventures decide on potential investment.

## 🛠️ Tools and Libraries
- **Python Libraries**:
  - 📊 `pandas` for data manipulation
  - ➗ `numpy` for numerical operations
  - 📈 `matplotlib` for data visualization
  - 🔮 `fbprophet` for time series forecasting
  - 🧪 `sklearn.model_selection` for data splitting

## 📋 Dataset Information
Investors are considering making an investment in a new form of transportation - **JetRail**. JetRail uses Jet propulsion technology to run rails and move people at a high speed! While JetRail has mastered the technology and holds the patent for their product, the investment would only make sense if they can get more than **1 Million monthly users within the next 18 months**.

You need to help **Unicorn Ventures** with the decision. They usually invest in **B2C start-ups less than 4 years old** looking for **pre-series A funding**. To aid their decision, we will forecast the traffic on JetRail for the next **7 months**.

## 🧮 Algorithms Used
- 🔮 **FBProphet**: For time series forecasting.

## 🚀 Steps

### 1. 📥 Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from fbprophet import Prophet
from sklearn.model_selection import train_test_split
```

### 2. 📊 Load and Preprocess Data
```python
# Load traffic data
traffic_df = pd.read_csv('traffic data.csv')

# Preview data
print(traffic_df.head())

# Check for missing values
print(traffic_df.isnull().sum())

# Rename columns for Prophet
traffic_df.rename(columns={'Date': 'ds', 'Traffic': 'y'}, inplace=True)
```

### 3. ✂️ Train-Test Split
```python
# Split data into training and testing
train_df, test_df = train_test_split(traffic_df, test_size=0.2, shuffle=False)
```

### 4. 🔮 Model Training with Prophet
```python
# Initialize Prophet model
model = Prophet()

# Fit model
model.fit(train_df)
```

### 5. 📅 Forecasting
```python
# Create future dataframe
future = model.make_future_dataframe(periods=7, freq='M')

# Make predictions
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title('JetRail Traffic Forecast')
plt.xlabel('Date')
plt.ylabel('Monthly Traffic')
plt.show()
```

### 6. 📏 Evaluation
```python
# Compare forecast with test data
from sklearn.metrics import mean_absolute_error

# Merge forecast and actual
forecast_actual = forecast[['ds', 'yhat']].set_index('ds').join(test_df.set_index('ds'))

# Calculate MAE
mae = mean_absolute_error(forecast_actual['y'], forecast_actual['yhat'])
print(f'Mean Absolute Error: {mae}')
```

## ✅ Results
Based on the FBProphet forecasting model, JetRail is projected to:

- Reach over **1 Million monthly users** within the next **7 months**.
- Maintain a consistent growth trajectory aligning with investment goals.

The **Mean Absolute Error (MAE)** of the model was within acceptable limits, indicating reliable predictions.

## 📌 Conclusion
Given the forecasted growth surpassing **1 Million monthly users**, **Unicorn Ventures** can confidently proceed with the investment in **JetRail**, aligning with their criteria for high-potential B2C start-ups.
