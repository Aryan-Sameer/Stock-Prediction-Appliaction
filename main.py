from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv("MicrosoftStock.csv")
# print(data.head()) # first 5 rows
# print(data.info()) # gives data type and other info about the data
# print(data.describe()) # gives the statistical info about the data

# plotting open and close prices
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['open'], label="Open", color="blue")
plt.plot(data['date'], data['close'], label="Close", color="red")
plt.title("Open-Close Price over Time")
plt.legend()
# plt.show()

# plotting volume to check outliers
plt.figure(figsize=(12,6))
plt.plot(data['date'], data['volume'], label="Volume", color="orange")
plt.title("Stock Volume over Time")
# plt.show()

# include only numeric columns
numeric_data = data.select_dtypes(include=["int64","float64"])

# creating heat map for correlation between features
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
# plt.show()

# converting the date column in to datetime data type
data['date'] = pd.to_datetime(data['date'])

# .loc() - select data by labels or conditions
prediction = data.loc[
    (data['date'] > datetime(2013,1,1)) &
    (data['date'] < datetime(2018,1,1))
]

plt.figure(figsize=(12,6))
plt.plot(data['date'], data['close'],color="blue")
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Price over time")

stock_close = data.filter(["close"])
dataset = stock_close.values #convert to numpy array
training_data_len = int(np.ceil(len(dataset) * 0.95)) # get 95% of the data set length


scaler = StandardScaler() # object from scikit learn to standardize data
scaled_data = scaler.fit_transform(dataset) # all features in the data are scaled to normal distribution

training_data = scaled_data[:training_data_len] #95% of the data

X_train, y_train = [], []

# sliding window for our stock (60 days)
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Sequential is a type of model where layers are added one after another in a linear stack.
model = keras.models.Sequential()

# Layer 1 - give all the data
# 64 memory cells / neurons, return_sequences=true outputs a sequence for each timestep
# input_shape=(timestamps, features) - timestamps - 60 days, 1 feature - closing price
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)))

# Layer 2 - final prediction - gives last sequence (representing learned features of the whole 60-day sequence)
model.add(keras.layers.LSTM(64, return_sequences=False))

# Layer 3 - to combine all features and relu introduces non linearity
model.add(keras.layers.Dense(128, activation="relu"))

# Layer 4 - avoid overfitting by dropping 50% of data
model.add(keras.layers.Dropout(0.5))

# Layer 5 - the predicted next day’s stock price with 1 neuron
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(X_train, y_train, epochs=20, batch_size=32)

# preparing the test data
test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1 ))

# Make a Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting data
train = data[:training_data_len]
test =  data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['date'], train['close'], label="Train (Actual)", color='blue')
plt.plot(test['date'], test['close'], label="Test (Actual)", color='orange')
plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
plt.title("Our Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# to calculate the accuracy of the model
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100

print(f"MAE: {mae}") # mean absolute error
print(f"RMSE: {rmse}") # Root Mean Squared Error
print(f"MAPE: {mape}%") # Mean Absolute Percentage Error