# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:04:19 2022

@author: kgavahi
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import copy
def MSE(X,Y):
	return np.mean((X-Y)**2)**.5
def CORREL(X,Y):
	X_bar = np.nanmean(X)
	Y_bar = np.nanmean(Y)
    
	numerator = np.nansum((X - X_bar) * (Y - Y_bar))
	denominator = np.nansum((X - X_bar)**2)**0.5 * np.nansum((Y - Y_bar)**2)**0.5
	r = numerator / denominator	
	return r
def KGE(x,y):
	r = CORREL(x,y)
	t1 = (r-1)**2
	t2 = (np.nanstd(x)/np.nanstd(y)-1)**2
	t3 = (np.nanmean(x)/np.nanmean(y)-1)**2
	return 1 - (t1+t2+t3)**0.5
def make_dataset(df):
    n_train = len(df)
    n_input = len(df) - (input_width + shift) + 1
    
    print('n_input', n_input)
    
    X = np.zeros([n_input, input_width, df.shape[1]])
    y = np.zeros([n_input, label_width, 1])
    for i in range(n_input):
    
        if i==0:
            print(np.arange(n_input)[i:i+input_width])
            print(np.arange(n_input)[i+input_width+(shift-label_width):i+input_width+shift])
    
        X[i, :, :] = df.iloc[i:i+input_width, :]
        
        
        
        y[i, :, 0] = df.iloc[i+input_width+(shift-label_width):i+input_width+shift, -1]
        
    
    ind = np.isnan(np.sum(y[:, :, 0], axis=1))
        
    y = y[~ind]
    X = X[~ind]

    ind = np.isnan(np.sum(X[:, :, 0], axis=1))
        
    y = y[~ind]
    X = X[~ind]


    print(y.shape)
    print(X.shape)        
    
    
    
    return X, y



data = pd.read_csv("08066500_chirps.txt", delimiter=',', header=0)

data['date'] = data['year'].astype(str) + \
    data['month'].astype(str).str.zfill(2) + \
        data['day'].astype(str).str.zfill(2)


data['time'] = pd.to_datetime(data['date'])

data.index= pd.to_datetime(data['time'])

data['daily_rain'] = data['rain1'] + data['rain2'] + data['rain3'] + data['rain4']



data = data[['PET', 'daily_rain', 'flow']]
#data = data[['flow']]

if data.shape[1]==1:
    mode = 'only flow'
else:
    mode = 'all three'

n = len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*0.85)]
test_df = data[int(n*0.85):]


'''
test_df.plot(label='test')
train_df.plot(label='train')
val_df.plot(label='val')
ax = plt.gca()
ax.set_ylabel('streamflow (cms)')
ax.legend()'''




#test_df.plot()

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std






input_width = 365
label_width = 1
shift = 1 


X_train, y_train = make_dataset(train_df)


X_val, y_val = make_dataset(val_df)
X_test, y_test = make_dataset(test_df)


X_train, y_train = shuffle(X_train, y_train, random_state=1)

tf.random.set_seed(7)




MAX_EPOCHS = 500

def compile_and_fit(model, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
  
  
  
  history = model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                      validation_data=(X_val, y_val),
                      callbacks=[early_stopping])
  
  return history

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(10, return_sequences=False),
    # Shape => [batch, time, features]
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1)
])



history = compile_and_fit(lstm_model)


y_pred =  lstm_model.predict(X_test)

y_pred_base =  X_test[:, -1, -1]


y_pred = y_pred.ravel()
y_test = y_test.ravel()


'''
print('test MSE:', MSE(y_pred, y_test))
print('test CORREL:', CORREL(y_pred, y_test))
print('test KGE:', KGE(y_pred, y_test))'''


print(mode)
print('test MSE:', MSE(y_pred, y_test), 'baseline MSE:', MSE(y_pred_base, y_test))
print('test CORREL:', CORREL(y_pred, y_test), 'baseline CORREL:', CORREL(y_pred_base, y_test))
print('test KGE:', KGE(y_pred, y_test), 'baseline KGE:', KGE(y_pred_base, y_test))


'''
n_input = len(test_df) - (input_width + shift) + 1
test_df[0+input_width+(shift-label_width):n_input+input_width+shift]['flow'].plot(marker = '*', markersize=4)

y_pred_df = copy.deepcopy(test_df)
y_pred_df.iloc[0+input_width+(shift-label_width):n_input+input_width+shift, -1] = y_pred
y_pred_df[0+input_width+(shift-label_width):n_input+input_width+shift]['flow'].plot(marker = '.', markersize=4)'''




aa
from scipy.stats import gaussian_kde

# Generate fake data
x = y_pred
y = y_test

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=1, cmap='jet')
plt.show()


#plt.plot(y_test)
#plt.plot(y_pred)
#plt.plot(y_pred_base)









