from sklearn.metrics import mean_squared_error
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
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
def weighted_mse(yTrue,yPred):
	from keras import backend as K
	ones = K.ones_like(yTrue[0,:]) # a simple vector with ones shaped as (10,)
	idx = K.cumsum(ones) # similar to a 'range(1,11)'

	return K.mean((1/idx)*K.square(yTrue-yPred))

def model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_epochs, batch_size, n_hidden, n_features, n_lags, scaler, last_values):
	from keras.layers import Dense, Activation, Dropout, LSTM
	from keras.models import Sequential
	from keras.optimizers import Adam

	drop_p = 0.05
	n_out = n_series

	model = Sequential()
	model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
	model.add(Dense(n_out, input_shape=(n_hidden,)))

	opt = Adam(lr=0.001, decay=0.0)
	model.compile(loss=weighted_mse, optimizer=opt)
	model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=0, shuffle=False)

	
	y_hat_val = model.predict(val_X)
	y_hat_test = model.predict(test_X)

	# predict last
	last = model.predict(np.expand_dims(last_values, axis=0))

	# for validation
	rmses_val = []
	rmse_val = 0
	weigth = 1.0
	step = 0.1
	for i in range(n_out):
		rmses_val.append(math.sqrt(mean_squared_error(val_y[:, i], y_hat_val[:, i])))
		rmse_val += rmses_val[-1]*weigth
		weigth -= step

	# for test
	rmses = []
	rmse = 0
	weigth = 1.5
	step = 0.1
	for i in range(n_out):
		rmses.append(math.sqrt(mean_squared_error(test_y[:, i], y_hat_test[:, i])))
		rmse += rmses[-1]*weigth
		weigth -= step

	# rmse = math.sqrt(mean_squared_error(y.reshape((len(y), -1)), y_hat.reshape((len(y), -1))))
	# rmse = np.sum([math.sqrt(mean_squared_error(y[i], y_hat[i])) for i in range(len(y))])
	# rmse = math.sqrt(mean_squared_error(y, y_hat))
	# rmse = math.sqrt(mean_squared_error(test_y, pred))

	# transform last values
	tmp = np.zeros((last.shape[1], n_features))
	tmp[:, 0] = last
	last = scaler.inverse_transform(tmp)[:, 0]

	return rmse, test_y, y_hat_test, last
	# return rmse, y, y_hat, last
	# return rmse, test_y, pred, last

############################ LSTM no slidding windows ################################
def model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_epochs, n_hidden, lr, n_features, n_lags, scaler, last_values):
	from keras.layers import LSTM
	from keras.models import Sequential
	from keras.optimizers import Adam
	import time

	drop_p = 0.05
	n_out = n_series
	lr_decay = 0.0

	model = Sequential()

	model.add(LSTM(n_hidden, input_shape=(None, train_X.shape[1]), return_sequences=True))
	model.add(LSTM(train_X.shape[1], return_sequences=True))

	opt = Adam(lr=lr, decay=lr_decay)
	model.compile(loss=weighted_mse, optimizer=opt)

	init = time.time()
	model.fit(np.expand_dims(train_X, axis=0), np.expand_dims(train_y, axis=0), epochs=n_epochs, verbose=0, shuffle=False)
	print('training time: ', time.time()-init)

	new_model = Sequential()
	new_model.add(LSTM(n_hidden, batch_input_shape=(1, None, train_X.shape[1]), return_sequences=True, stateful=True))
	new_model.add(LSTM(train_X.shape[1], return_sequences=False, stateful=True))

	# Transfer learning
	new_model.set_weights(model.get_weights())

	# # Validation
	# preds_val = []
	# obs_val = []
	# for i in range(0, len(val_y), 10):
	# 	pred1 = new_model.predict(np.expand_dims(np.append(train_X, val_X[:i], axis=0), axis=0))
	# 	pred2 = new_model.predict(np.expand_dims(pred1, axis=0))
	# 	pred3 = new_model.predict(np.expand_dims(pred2, axis=0))
	# 	pred4 = new_model.predict(np.expand_dims(pred3, axis=0))
	# 	pred5 = new_model.predict(np.expand_dims(pred4, axis=0))
	# 	pred6 = new_model.predict(np.expand_dims(pred5, axis=0))
	# 	pred7 = new_model.predict(np.expand_dims(pred6, axis=0))
	# 	pred8 = new_model.predict(np.expand_dims(pred7, axis=0))
	# 	pred9 = new_model.predict(np.expand_dims(pred8, axis=0))
	# 	pred10 = new_model.predict(np.expand_dims(pred9, axis=0))

	# 	preds_val.append([pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0], pred5[0][0], pred6[0][0], pred7[0][0], pred8[0][0], pred9[0][0], pred10[0][0]])
	# 	obs_val.extend(val_y[i:i+10])

	# Testing
	preds_test = []
	obs_test = []
	for i in range(0, len(test_y), 10):
		pred1 = new_model.predict(np.expand_dims(np.append(np.append(train_X, val_X, axis=0), test_X[:i], axis=0), axis=0))
		pred2 = new_model.predict(np.expand_dims(pred1, axis=0))
		pred3 = new_model.predict(np.expand_dims(pred2, axis=0))
		pred4 = new_model.predict(np.expand_dims(pred3, axis=0))
		pred5 = new_model.predict(np.expand_dims(pred4, axis=0))
		pred6 = new_model.predict(np.expand_dims(pred5, axis=0))
		pred7 = new_model.predict(np.expand_dims(pred6, axis=0))
		pred8 = new_model.predict(np.expand_dims(pred7, axis=0))
		pred9 = new_model.predict(np.expand_dims(pred8, axis=0))
		pred10 = new_model.predict(np.expand_dims(pred9, axis=0))

		if(len(test_y) - i >= n_series):
			preds_test.append([pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0], pred5[0][0], pred6[0][0], pred7[0][0], pred8[0][0], pred9[0][0], pred10[0][0]])
			obs_test.append(test_y[i:i+10, 0])

	preds_test = np.array(preds_test)
	obs_test = np.array(obs_test)

	# for test
	rmses = []
	rmse = 0
	weigth = 1.5
	step = 0.1
	for i in range(n_out):
		rmses.append(math.sqrt(mean_squared_error(obs_test[:, i], preds_test[:, i])))
		rmse += rmses[-1]*weigth
		weigth -= step

	last = [new_model.predict(np.expand_dims(np.expand_dims(last_values, axis=0), axis=0))]
	for i in range(n_series - 1):
		last.append(new_model.predict(np.expand_dims(last[-1], axis=0)))
	return rmse, obs_test, preds_test, np.array(last)[:, :, 0].ravel()

###################### random forest ##########################
def model_random_forest(train_X, test_X, train_y, test_y, n_series, n_estimators, max_features, min_samples, n_features, n_lags, scaler, last_values):
	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples, n_jobs=-1)
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