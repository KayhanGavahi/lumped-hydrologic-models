# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:04:19 2022

@author: kgavahi
"""

import pandas as pd
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import dataretrieval.nwis as nwis
import copy
def MSE(X,Y):
	return np.mean((X-Y)**2)**.5
def NSE(X,Y):
	
    return 1 - np.sum(np.abs(X-Y)) / np.sum(np.abs(Y - np.mean(Y)))
    
    
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
    n_input = n_train - (input_width + shift) + 1
    
    print('n_input', n_input)
    
    #X = np.zeros([n_input, input_width, df.shape[1]])
    X = np.zeros([n_input, input_width])
    y = np.zeros([n_input, label_width])
    for i in range(n_input):
    
        if i==0:
            print(np.arange(n_input)[i:i+input_width])
            print(np.arange(n_input)[i+input_width+(shift-label_width):i+input_width+shift])
    
        #X[i, :, :] = df.iloc[i:i+input_width, :]
        X[i, :] = df.iloc[i:i+input_width, 0]
        
        
        
        y[i, :] = df.iloc[i+input_width+(shift-label_width):i+input_width+shift, -1]
        
    
    ind = np.isnan(np.sum(y[:, :], axis=1))
        
    y = y[~ind]
    X = X[~ind]

    ind = np.isnan(np.sum(X[:, :], axis=1))
        
    y = y[~ind]
    X = X[~ind]


    print(y.shape)
    print(X.shape)        
    
    
    
    return X, y



#data = pd.read_csv("08066500_nldas.txt", delimiter=',', header=0)
#data = pd.read_csv("08066500.txt", delimiter=',', header=0)

site = '02359170'

data = nwis.get_record(sites=site, start='2000-01-01', end='2020-12-31', parameterCd='00060', service='dv') # unit = cfs
data = data[['00060_Mean']]

n = len(data)
train_df = data[0:int(n*0.6)]
val_df = data[int(n*0.6):int(n*0.8)]
test_df = data[int(n*0.8):]




train_df['00060_Mean'].plot(label='train', c='#1f77b4')
val_df['00060_Mean'].plot(label='val', c='g')
test_df['00060_Mean'].plot(label='test', c='darkorange')
ax = plt.gca()
ax.set_ylabel('streamflow (cfs)')
ax.legend()

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


input_width = 7
label_width = 1
shift = 1


X_train, y_train = make_dataset(train_df)
X_val, y_val = make_dataset(val_df)
X_test, y_test = make_dataset(test_df)


X_train, y_train = shuffle(X_train, y_train, random_state=1)


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(max_depth=100, random_state=0)
forest.fit(X_train.reshape(X_train.shape[0], input_width), 
         y_train.reshape(y_train.shape[0], 1).ravel())



y_pred_rf = forest.predict(X_test.reshape(X_test.shape[0],
                                        input_width))* train_std['00060_Mean'] + train_mean['00060_Mean']

#from sklearn.svm import SVR
#svr_m = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
#y_pred_svr = svr_m.fit(X_train, y_train.ravel()).predict(X_test)



y_test = y_test.ravel() * train_std['00060_Mean'] + train_mean['00060_Mean']


import hydroeval as he
kge, r, alpha, beta = he.evaluator(he.kge, y_pred_rf, y_test)
RMSE = np.sqrt(np.mean((y_pred_rf-y_test)**2))*0.0283168
print('KGE = ', kge)
#kge, r, alpha, beta = he.evaluator(he.kge, y_pred_svr, y_test)


feature_names = [f"day t-{input_width-i}" for i in range(X_train.shape[1])]
importances = forest.feature_importances_
forest_importances = pd.Series(importances, index=feature_names)
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


aa

data = data.rename(columns={0: "year", 1: "month", 2: "day", 3: "flow", 4: "PET", 
                            5: "rain1", 6: "rain2", 7: "rain3", 8: "rain4"})


data['date'] = data['year'].astype(str) + \
    data['month'].astype(str).str.zfill(2) + \
        data['day'].astype(str).str.zfill(2)


data['time'] = pd.to_datetime(data['date'])

data.index= pd.to_datetime(data['time'])

data['daily_rain'] = data['rain1'] + data['rain2'] + data['rain3'] + data['rain4']



data = data[['PET', 'daily_rain', 'flow']]
data[['flow']] = data[['flow']] * 35.314666212661

#data.to_csv('08066500_nldas.csv')

if data.shape[1]==1:
    mode = 'only flow'
else:
    mode = 'all three'

n = len(data)
train_df = data[0:int(n*0.6)]
val_df = data[int(n*0.6):int(n*0.8)]
test_df = data[int(n*0.8):]




train_df['flow'].plot(label='train', c='#1f77b4')
val_df['flow'].plot(label='val', c='g')
test_df['flow'].plot(label='test', c='darkorange')
ax = plt.gca()
ax.set_ylabel('streamflow (cfs)')
ax.legend()

#plt.savefig('train_test_val.png', dpi=600)
#### See this for Harvey:
#### https://www.weather.gov/crp/hurricane_harvey

print('max flow at row number = ', data['flow'].argmax(axis=0))
print(data.iloc[6085])



train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std






input_width = 50
label_width = 1
shift = 1


X_train, y_train = make_dataset(train_df)
X_val, y_val = make_dataset(val_df)
X_test, y_test = make_dataset(test_df)


X_train, y_train = shuffle(X_train, y_train, random_state=1)



from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=100, random_state=0)
regr.fit(X_train.reshape(X_train.shape[0], input_width*X_train.shape[-1]), 
         y_train.reshape(y_train.shape[0], 1).ravel())

tf.random.set_seed(7)







MAX_EPOCHS = 50

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
y_pred_rf = regr.predict(X_test.reshape(X_test.shape[0],
                                        input_width*X_test.shape[-1]))* train_std['flow'] + train_mean['flow']


y_pred_base =  X_test[:, -1, -1] * train_std['flow'] + train_mean['flow']


y_pred = y_pred.ravel() * train_std['flow'] + train_mean['flow']
y_test = y_test.ravel() * train_std['flow'] + train_mean['flow']



'''
print(mode)
print('test MSE RF:', MSE(y_pred_rf, y_test), 'baseline MSE:', MSE(y_pred_base, y_test))
print('test CORREL RF:', CORREL(y_pred_rf, y_test), 'baseline CORREL:', CORREL(y_pred_base, y_test))
print('test KGE RF:', KGE(y_pred_rf, y_test), 'baseline KGE:', KGE(y_pred_base, y_test))'''


print(mode)
print('test MSE LSTM:', NSE(y_pred, y_test), 'baseline MSE:', NSE(y_pred_base, y_test))
print('test CORREL LSTM:', CORREL(y_pred, y_test), 'baseline CORREL:', CORREL(y_pred_base, y_test))
print('test KGE LSTM:', KGE(y_pred, y_test), 'baseline KGE:', KGE(y_pred_base, y_test))




plt.figure()
n_input = len(test_df) - (input_width + shift) + 1

y_pred_df = copy.deepcopy(test_df)

test_df = test_df.rename(columns={"flow": "Obs"})
test_df['Obs'] = test_df['Obs'] * train_std['flow'] + train_mean['flow']

test_df[0+input_width+(shift-label_width):n_input+input_width+shift]['Obs'].plot(marker = '*', markersize=4)




y_pred_df.iloc[0+input_width+(shift-label_width):n_input+input_width+shift, -1] = y_pred

y_pred_df = y_pred_df.rename(columns={"flow": "LSTM"})
y_pred_df[0+input_width+(shift-label_width):n_input+input_width+shift]['LSTM'].plot(marker = '.', markersize=4)


ax = plt.gca()
ax.set_ylabel('streamflow (cfs)')
ax.legend()

plt.savefig('7_day_ahead_bl.png', dpi=600, bbox_inches='tight')

#.savefig(f'MaxEpoch_{MAX_EPOCHS}.png')


'''
test_df['LSTM'] = y_pred_df['LSTM'] 



product = np.zeros([1388, 102])



product[:, 0] = np.array(test_df['Obs'])

for i in range(1388):

    #product[i, 2:] = np.random.randn(100) * np.array(test_df['LSTM'])[i] * .05 + np.array(test_df['LSTM'])[i]
    
    
    product[i, 1] = np.mean(product[i, 2:103])
    
product[input_width:, 1] = y_pred_base

for j in range(100):
    
    product[input_width:, j+2] = y_pred_base
    


Product = np.genfromtxt('08075770.txt', delimiter=',')

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
#plt.plot(y_pred_base)'''









