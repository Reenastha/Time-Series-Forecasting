# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:37:15 2023

@author: reena
"""

#Load the Libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split

# Read the data
path = "C:/Users/reena/OneDrive - Lamar University/Desktop/Machine learning/TimeSeries_class"
os.chdir(path)
df = pd.read_csv('J17_KL.csv')

# Change the index to date format
df.index = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Visualize the data
df["WaterLevelElevation"].plot(figsize=(16,8) , color='blue')
plt.title("WaterLevelElevation(J17)")
plt.xlabel('Date', fontsize=18)
plt.ylabel('WaterLevel', fontsize = 18)

# Choose the Prediction column
df1 = df["WaterLevelElevation"]
df1 = pd.DataFrame(df1)

# Take data from the dataframe and return a numpy array
df2 = df1.values
df2.shape

#Normalizing the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df2)

#Split the data to training and test sets
# 70% to Train , 30% to Test
train_size = int(len(df2)*.70)
test_size = len(df2) - train_size

print("Train Size :",train_size,"Test Size :",test_size)
train_data = scaled_data[ :train_size , 0:1 ]
test_data = scaled_data[ train_size-10: , 0:1 ]

train_data.shape,test_data.shape

# Creating the Train Set
# Creating a Training set with 10 time-steps and 1 output
time_steps = 10
x_train = []
y_train = []

for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping the input to three-dimensional array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train.shape , y_train.shape

# Building the LSTM Model
model =Sequential()
#Add LSTM Layer with 50 neurons and return sequences
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
#Add LSTM Layer with 64 neurons and return single output
model.add(LSTM(64, return_sequences= False))
#Add Dense layer with 16 and 1 neuron
model.add(Dense(16))
model.add(Dense(1))

# Compile the LSTM Model
model.compile(optimizer = 'adam', loss = 'mse' , metrics="mean_absolute_error")

#Summary of the model
model.summary()

# Fitting the LSTM to the Training set
callbacks = [EarlyStopping(monitor='loss', patience=10 , restore_best_weights=True)]
history = model.fit(x_train, y_train, epochs = 30, batch_size = 64 , callbacks = callbacks )

# Visualizing the performance of the model
plt.plot(history.history["loss"])
plt.plot(history.history["mean_absolute_error"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("Losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

# Creating the Test Set
# Creating a testing set with 10 time-steps and 1 output
x_test = []
y_test = []

for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, 0])
    y_test.append(test_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_test.shape , y_test.shape

# Predicting the testing dataset
# inverse y_test scaling
predictions = model.predict(x_test)
predictions.shape
#inverse predictions scaling to original scale
predictions = scaler.inverse_transform(predictions)
predictions.shape

# Visualizing the Prediction with Data
train = df1.iloc[:train_size , 0:1]
test = df1.iloc[train_size: , 0:1]
test['Predictions'] = predictions

# Plot the training, test and prediction dataset vs date
plt.figure(figsize=(16,6))
plt.title('Water level Elevation prediction(J17)' , fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('WaterLevelElevation' ,fontsize=18)
plt.plot(train['WaterLevelElevation'],linewidth=3)
plt.plot(test['WaterLevelElevation'],linewidth=3)
plt.plot(test["Predictions"],linewidth=3)
plt.legend(['Train','Test','Predictions'])

#Plot the test and predicton dataset vs date
plt.figure(figsize=(16,6))
plt.title('Water level Elevation prediction(J17)' , fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylabel('WaterLevelElevation' ,fontsize=18)
plt.plot(test['WaterLevelElevation'],linewidth=3)
plt.plot(test["Predictions"],linewidth=3)
plt.legend(['Test','Predictions'])

#See test and predicted values
test.head()




