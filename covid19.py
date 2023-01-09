#%%
# 1. Import packages
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import re
import os

#%%
# 2. Data Loading

CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv')
covid_df = pd.read_csv(CSV_PATH)

TEST_CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')

test_df = pd.read_csv(TEST_CSV_PATH)


#%%
# 3. Data Inspection / Data Visualization
covid_df.head(10)
covid_df.tail(10)
covid_df.describe().T
print(covid_df.info())
print(covid_df.isna().sum())
print(covid_df.isnull().sum())
print(covid_df.duplicated().sum())


#%%
# 4. Data Cleaning
# Since we see the type of cases_new in object form, we need to convert it into numerical, also, from column 24 to column 30 have around 345 nan values.

clusters = ['cluster_import','cluster_religious','cluster_community',	'cluster_highRisk',	'cluster_education',	'cluster_detentionCentre',	'cluster_workplace']

covid_df['cases_new'] = pd.to_numeric(covid_df['cases_new'], errors='coerce')
covid_df.info()

plt.figure(figsize=(8, 8))
plt.plot(covid_df['cases_new'].values)
plt.show()


#%%
# Method 2) Filling NAN
temp1 = covid_df['cases_new'].fillna(covid_df['cases_new'].mean())
temp1 = covid_df['cases_new'].fillna(covid_df['cases_new'].median())
temp1 = covid_df['cases_new'].fillna(method='ffill')
plt.plot(temp1)
plt.show()

covid_df[clusters] = covid_df[clusters].fillna(0)


#%%
# Method 3) Interpolation
covid_df.info()
covid_df.isna().sum()

covid_df['cases_new'] = covid_df['cases_new'].interpolate(method='polynomial', order=2)

print(covid_df.isna().sum())
print(covid_df.info())
plt.figure(figsize=(10,10))
plt.plot(covid_df['cases_new'].values)
plt.show()


#%%
# 4. Features Selection

new_cases = covid_df['cases_new'].values


#%%
# 5. Data Preprocessing"""

# mms = MinMaxScaler()
# # Method 1) Reshaping into the right shape
# new_cases = mms.fit_transform(new_cases[::,None])

X = []
y = []
win_size = 30
# for i in range(len(open)-8):
#   X.append(open[i:i+7])
#   y.append(open[i+7])
#   print('x: {}'.format(open[i:i+7]))
#   print('y: {}'.format(open[i+8]))

for i in range(win_size, len(new_cases)):
  X.append(new_cases[i-win_size:i])
  y.append(new_cases[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#%%
# 6. Model deployment

model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1:]),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='mspe', metrics=['mse', 'mape'])


#%%
# Tensorboad callback
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

ts = TensorBoard(log_dir=LOGS_PATH)
es = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

model.fit(X_train,y_train,epochs=10,batch_size=64,callbacks=[ts,es])


# %%
model.save('model.h5')