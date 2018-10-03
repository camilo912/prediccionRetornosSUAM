from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

import numpy as np
import math

def calculate_rmse(n_series, n_features, n_lags, X, y, scaler, model):
	yhat = model.predict(X)
	Xs = np.ones((X.shape[0], n_lags * n_features))
	yhat = yhat.reshape(-1, n_series)
	#print(yhat[:, 0].reshape(-1, 1).shape)
	#print(Xs[:, -(n_features - 1):].shape)
	#print(Xs[:, -(n_features - n_series):].shape)
	inv_yhats = []
	for i in range(n_series):
		inv_yhat = np.concatenate((Xs[:, -(n_features - 1):], yhat[:, i].reshape(-1, 1)), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, -1]
		inv_yhats.append(inv_yhat)

	inv_yhat = np.array(inv_yhats).T
	
	# invert scaling for actual
	y = y.reshape((len(y), n_series))
	inv_ys = []
	for i in range(n_series):
		inv_y = np.concatenate((Xs[:, -(n_features - 1):], y[:, i].reshape(-1, 1)), axis=1)
		inv_y = scaler.inverse_transform(inv_y)
		inv_y = inv_y[:,-1]
		inv_ys.append(inv_y)

	inv_y = np.array(inv_ys).T


	# calculate RMSE	
	rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat)) / np.max(inv_y) * 100
	return inv_y, inv_yhat, rmse

def predict_last(n_series, n_features, n_lags, X, scaler, model, dim):
	if(dim):
		X = np.expand_dims(X, axis=0)
		# X = X.reshape(1, n_lags, -1) # only for dim, only for LSTM or RNN
	yhat = model.predict(X)
	if(not dim):
		yhat = yhat.reshape(-1, 1) # only for no dim
	Xs = np.ones((X.shape[0], n_lags * n_features))
	# # inv_yhat = np.concatenate((yhat, Xs[:, -(n_features - n_series):]), axis=1)
	# inv_yhat = np.concatenate((yhat, Xs[:, -(n_features - n_series):]), axis=1)
	# inv_yhat = scaler.inverse_transform(inv_yhat)

	# inv_yhat = inv_yhat[:,0:n_series]
	inv_yhats = []
	for i in range(n_series):
		inv_yhat = np.concatenate((Xs[:, -(n_features - 1):], yhat[:, i].reshape(-1, 1)), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, -1]
		inv_yhats.append(inv_yhat)

	inv_yhat = np.array(inv_yhats).T
	#return inv_yhat[-1]
	return inv_yhat

############################# LSTM ####################################
def model_lstm(train_X, test_X, train_y, test_y, n_series, n_epochs, batch_size, lr, n_hidden, n_features, n_lags, scaler, last_values):
	# from keras.layers import Dense, Activation, Dropout, LSTM
	# from keras.models import Sequential

	# drop_p = 0.05
	# n_out = n_series

	# model = Sequential()
	# model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
	# #model.add(Dropout(drop_p))
	# #model.add(LSTM(n_hidden, input_shape=(None, None), activation='relu'))
	# #model.add(Dropout(drop_p))
	# #model.add(Dense(n_features, activation='linear'))
	# model.add(Dense(n_features))

	# model.compile(loss='mse', optimizer='adam')
	# #print(train_X.shape)
	# #print(train_y.shape)
	# model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=0)

	# # predict testing
	# y = []
	# y_hat = []
	# # for i in range(len(test_X) - n_out):
	# for i in range(test_X):
	# 	x = test_X[i]
	# 	#print(x.shape)
	# 	#raise Exception('Debug')
	# 	v_arr = []
	# 	pred_arr = []
	# 	for j in range(n_out):
	# 		pred = model.predict(np.expand_dims(x, axis=0))
	# 		v_arr.append(test_y[i])#[0])	
	# 		pred_arr.append(pred[-1])#[0])
	# 		x = np.roll(x, -1, axis=0)
	# 		x[-1, :] = pred[-1]
	# 	y.append(v_arr)
	# 	y_hat.append(pred_arr)

	# y = np.array(y)
	# y = y.reshape(y.shape[:2])
	# y_hat = np.array(y_hat)
	# y_hat = y_hat.reshape(y_hat.shape[:2])

	# # predict last
	# x = last_values
	# last = []
	# for j in range(n_out):
	# 		pred = model.predict(np.expand_dims(x, axis=0))
	# 		last.append(pred[-1])#[0])
	# 		x = np.roll(x, -1, axis=0)
	# 		x[-1, :] = pred[-1]

	# last = np.array(last).reshape(n_out)
	# rmse = math.sqrt(mean_squared_error(y, y_hat))

	# # print(last)
	# # transform last values
	# tmp = np.zeros((len(last), n_features))
	# tmp[:, 0] = last
	# last = scaler.inverse_transform(tmp)[:, 0]

	# # tempo = np.zeros((1, n_features))
	# # tempo0 = scaler.inverse_transform(tempo)
	# # tempo[0, 0] = 1
	# # tempo1 = scaler.inverse_transform(tempo)
	# # print(tempo0)
	# # print(tempo1)

	# return rmse, y, y_hat, last

	from keras.layers import Dense, Activation, Dropout, LSTM
	from keras.models import Sequential

	drop_p = 0.05
	n_out = n_series

	model = Sequential()
	model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
	model.add(Dense(n_out, input_shape=(n_hidden,)))

	model.compile(loss='mse', optimizer='adam')
	model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=0)

	pred = model.predict(test_X)


	# predict last
	last = model.predict(np.expand_dims(last_values, axis=0))

	last = np.array(last).reshape(n_out)
	rmse = math.sqrt(mean_squared_error(test_y, pred))

	# transform last values
	tmp = np.zeros((len(last), n_features))
	tmp[:, 0] = last
	last = scaler.inverse_transform(tmp)[:, 0]

	# return rmse, y, y_hat, last
	return rmse, test_y, pred, last



###################### random forest ##########################
def model_random_forest(train_X, test_X, train_y, test_y, n_series, n_estimators, max_features, min_samples, n_features, n_lags, scaler, last_values):
	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples, n_jobs=4)
	model.fit(train_X, train_y.ravel())

	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, y, y_hat, last

####################### ada boost ###############################
def model_ada_boost(train_X, test_X, train_y, test_y, n_series, n_estimators, lr, n_features, n_lags, scaler, last_values):
	from sklearn.ensemble import AdaBoostRegressor
	model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=lr)
	model.fit(train_X, train_y.ravel())

	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, y, y_hat, last

####################################### SVM ##############################
def model_svm(train_X, test_X, train_y, test_y, n_series, n_features, n_lags, scaler, last_values):
	from sklearn.svm import SVR
	model = SVR(kernel='poly', degree=1)
	model.fit(train_X, train_y.ravel())

	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, y, y_hat, last

###################################### ARIMA #########################################
def model_arima(train_X, test_X, train_y, test_y, n_series, d, q, n_features, n_lags, scaler, last_values):
	#from statsmodels.tsa.arima_model import ARIMA
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	import warnings
	test_size = len(test_y)
	test_y = np.append(test_y, last_values)
	y_hat = []
	for i in range(test_size + 1):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ConvergenceWarning)
			model = SARIMAX(train_X, order=(n_lags, d, q))
			model_fit = model.fit(disp=0, maxiter=200, method='powell')
			output = model_fit.forecast()[0]
			y_hat.append(output)
			train_X = np.append(train_X, test_y[i])
	#print(test_y)
	#print(y_hat)
	rmse = math.sqrt(mean_squared_error(test_y, y_hat))
	model = SARIMAX(train_X, order=(n_lags, d, q))
	model_fit = model.fit(disp=0)
	last = model_fit.forecast()[0]
	last = last.reshape(-1, 1)
	Xs = np.ones((last.shape[0], n_lags * n_features))
	# inv_yhat = np.concatenate((yhat, Xs[:, -(n_features - n_series):]), axis=1)
	inv_yhat = np.concatenate((last, Xs[:, -(n_features - n_series):]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)

	inv_yhat = inv_yhat[:,0:n_series]

	return rmse, test_y, y_hat, inv_yhat[-1]