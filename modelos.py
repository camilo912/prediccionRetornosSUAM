from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import numpy as np
import math

def calculate_rmse(n_series, n_features, n_lags, X, y, scaler, model):
	yhat = model.predict(X)
	Xs = np.ones((X.shape[0], n_lags * n_features))
	yhat = yhat.reshape(-1, n_series)
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
	rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat)) # / np.max(inv_y) * 100
	return inv_y, inv_yhat, rmse

def predict_last(n_series, n_features, n_lags, X, scaler, model, dim):
	if(dim):
		X = np.expand_dims(X, axis=0)
		# X = X.reshape(1, n_lags, -1) # only for dim, only for LSTM or RNN
	yhat = model.predict(X)
	if(not dim):
		yhat = yhat.reshape(-1, 1) # only for no dim
	Xs = np.ones((X.shape[0], n_lags * n_features))

	inv_yhats = []
	for i in range(n_series):
		inv_yhat = np.concatenate((Xs[:, -(n_features - 1):], yhat[:, i].reshape(-1, 1)), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, -1]
		inv_yhats.append(inv_yhat)

	inv_yhat = np.array(inv_yhats).T
	return inv_yhat

############################# LSTM ####################################
def weighted_mse(yTrue, yPred):
	from keras import backend as K
	ones = K.ones_like(yTrue[0,:]) # a simple vector with ones shaped as (10,)
	idx = K.cumsum(ones) # similar to a 'range(1,11)'

	return K.mean((1/idx)*K.square(yTrue-yPred))

def model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_epochs, batch_size, n_hidden, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity):
	from keras.layers import Dense, Activation, Dropout, LSTM
	from keras.models import Sequential
	from keras.optimizers import Adam

	drop_p = 0.05
	n_out = n_series
	verbose = 0 if verbosity < 2 else min(verbosity - 1, 2)

	model = Sequential()
	model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
	model.add(Dense(n_out, input_shape=(n_hidden,)))

	opt = Adam(lr=0.001, decay=0.0)
	model.compile(loss=weighted_mse, optimizer=opt)
	model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
	
	y_hat_val = model.predict(val_X)
	y_hat_test = model.predict(test_X)

	# predict last
	last = model.predict(np.expand_dims(last_values, axis=0))
	if(calc_val_error):
		# for validation
		rmses_val = []
		rmse_val = 0
		weigth = 1.0
		step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(y_hat_val[:, i].ravel()), n_features))
			tmp[:, 0] = y_hat_val[:, i].ravel()
			tmp = scaler.inverse_transform(tmp)[:, 0]

			rmses_val.append(math.sqrt(mean_squared_error(val_y[:, i], tmp)))
			rmse_val += rmses_val[-1]*weigth
			weigth -= step
	else:
		rmse_val = None

	if(calc_test_error):
		# for test
		rmses = []
		rmse = 0
		weigth = 1.5
		step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(y_hat_test[:, i].ravel()), n_features))
			tmp[:, 0] = y_hat_test[:, i].ravel()
			tmp = scaler.inverse_transform(tmp)[:, 0]

			rmses.append(math.sqrt(mean_squared_error(test_y[:, i], tmp)))
			rmse += rmses[-1]*weigth
			weigth -= step
	else:
		rmse = None

	# transform last values
	tmp = np.zeros((last.shape[1], n_features))
	tmp[:, 0] = last
	last = scaler.inverse_transform(tmp)[:, 0]

	return rmse, rmse_val, test_y, y_hat_test, last

############################ LSTM no slidding windows ################################
def model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_epochs, lr, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity):
	from keras.layers import LSTM, Input
	from keras.models import Sequential, Model
	from keras.optimizers import Adam
	import time

	drop_p = 0.05
	n_out = n_series
	lr_decay = 0.0
	verbose = 0 if verbosity < 2 else min(verbosity - 1, 2)

	train_model = Sequential()

	train_model.add(LSTM(train_X.shape[1], batch_input_shape=(1, None, train_X.shape[1]), return_sequences=True, stateful=True))

	opt = Adam(lr=lr, decay=lr_decay)
	train_model.compile(loss=weighted_mse, optimizer=opt)

	for epoch in range(n_epochs):
		train_model.fit(np.expand_dims(train_X, axis=0), np.expand_dims(train_y, axis=0), validation_data=(np.expand_dims(val_X, axis=0), np.expand_dims(val_y, axis=0)),epochs=1, verbose=verbose, shuffle=False, batch_size=1)
		train_model.reset_states()

	main_input = Input(batch_shape=(1, None, train_X.shape[1]))

	lstm_out = LSTM(train_X.shape[1], batch_input_shape=(1, None, train_X.shape[1]), return_sequences=True, stateful=True, return_state=True, name='my_lstm')(main_input)

	model = Model(inputs=[main_input], outputs=[lstm_out[0], lstm_out[1], lstm_out[2]])

	model.set_weights(train_model.get_weights())

	if(calc_val_error):
		# Validation
		preds_val = []
		obs_val = []
		layer = model.get_layer(name='my_lstm')
		model.reset_states()
		_, sh, sc = model.predict(np.expand_dims(train_X, axis=0))
		for i in range(0, len(val_y)):
			model.reset_states()
			layer.reset_states(states=(sh, sc))
			preds, sh, sc = model.predict(val_X[i].reshape(1, 1, n_features))
			preds = preds[-1][-1].reshape(-1, n_features)
			for j in range(n_out - 1):
				preds = np.append(preds, model.predict(preds[-1].reshape(1, 1, n_features))[0][-1][-1].reshape(-1, n_features), axis=0)

			if(len(val_y) - i >= n_out):
				preds_val.append(preds[:, 0])
				obs_val.append(val_y[i:i+n_out, 0])

		preds_val = np.array(preds_val)
		obs_val = np.array(obs_val)

		# for validation
		rmses_val = []
		rmse_val = 0
		weigth = 1.5
		step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(preds_val[:, i].ravel()), n_features))
			tmp[:, 0] = preds_val[:, i].ravel()
			tmp = scaler.inverse_transform(tmp)[:, 0]

			rmses_val.append(math.sqrt(mean_squared_error(obs_val[:, i], tmp)))
			rmse_val += rmses_val[-1]*weigth
			weigth -= step
	else:
		rmse_val = None


	# Testing
	preds_test = []
	obs_test = []
	layer = model.get_layer(name='my_lstm')
	model.reset_states()
	_, sh, sc = model.predict(np.expand_dims(np.append(train_X, val_X, axis=0), axis=0))
	for i in range(0, len(test_y)):
		model.reset_states()
		layer.reset_states(states=(sh, sc))
		preds, sh, sc = model.predict(test_X[i].reshape(1, 1, n_features))
		preds = preds[-1][-1].reshape(-1, n_features)
		for j in range(n_out - 1):
			preds = np.append(preds, model.predict(preds[-1].reshape(1, 1, n_features))[0][-1][-1].reshape(-1, n_features), axis=0)

		if(len(test_y) - i >= n_out):
			preds_test.append(preds[:, 0])
			obs_test.append(test_y[i:i+n_out, 0])

	preds_test = np.array(preds_test)
	obs_test = np.array(obs_test)

	if(calc_test_error):
		# for test
		rmses = []
		rmse = 0
		weigth = 1.5
		step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(preds_test[:, i].ravel()), n_features))
			tmp[:, 0] = preds_test[:, i].ravel()
			tmp = scaler.inverse_transform(tmp)[:, 0]

			rmses.append(math.sqrt(mean_squared_error(obs_test[:, i], tmp)))
			rmse += rmses[-1]*weigth
			weigth -= step
	else:
		rmse = None

	full_data = np.append(np.append(np.append(train_X, val_X, axis=0), test_X, axis=0), np.expand_dims(last_values, axis=0), axis=0)

	model.reset_states()
	last = []
	for i in range(n_series):
		pred, _, _ = model.predict(np.expand_dims(full_data, axis=0))
		pred = pred[0].reshape(-1, n_features)
		full_data = np.expand_dims(pred[-1], axis=0) # np.append(full_data, pred[-1].reshape(1, -1), axis=0)
		last.append(pred[-1 ,0])

	last = np.array(last).reshape(1, -1)

	tmp = np.zeros((last.shape[1], n_features))
	tmp[:, 0] = last
	last = scaler.inverse_transform(tmp)[:, 0]

	return rmse, rmse_val, obs_test, preds_test, last.ravel()

###################### random forest ##########################
def model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_estimators, max_features, min_samples, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity):
	from sklearn.ensemble import RandomForestRegressor

	verbose = 0 if verbosity < 2 else verbosity - 1
	model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples, n_jobs=-1, verbose=verbose)
	model.fit(train_X, train_y.ravel())
	if(calc_val_error):
		_, _, rmse_val = calculate_rmse(n_series, n_features, n_lags, val_X, val_y, scaler, model)
	else:
		rmse_val=None

	if(calc_test_error):
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
	else:
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
		rmse = None

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, rmse_val, y, y_hat, last

####################### ada boost ###############################
def model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_estimators, lr, max_depth, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity):
	from sklearn.ensemble import AdaBoostRegressor
	from sklearn.tree import DecisionTreeRegressor

	model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators, learning_rate=lr)
	model.fit(train_X, train_y.ravel())
	
	if(calc_val_error):
		_, _, rmse_val = calculate_rmse(n_series, n_features, n_lags, val_X, val_y, scaler, model)
	else:
		rmse_val=None

	if(calc_test_error):
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
	else:
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
		rmse = None
	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, rmse_val, y, y_hat, last

####################################### SVM ##############################
def model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity):
	from sklearn.svm import SVR

	verbose = 0 if verbosity < 2 else verbosity - 1
	model = SVR(kernel='poly', degree=1, gamma='scale', verbose=verbose)
	model.fit(train_X, train_y.ravel())

	if(calc_val_error):
		_, _, rmse_val = calculate_rmse(n_series, n_features, n_lags, val_X, val_y, scaler, model)
	else:
		rmse_val=None

	if(calc_test_error):
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
	else:
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
		rmse = None

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, rmse_val, y, y_hat, last

###################################### ARIMA #########################################
def model_arima(train_X, val_X, test_X, train_y, val_y, test_y, n_series, d, q, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity):
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	from numpy.linalg.linalg import LinAlgError
	import warnings

	verbose = 0 if verbosity < 2 else verbosity - 1
	test_size = len(test_y)
	test_y = np.append(test_y, last_values)
	y_hat = []
	y_hat_val = []
	try:

		for i in range(len(val_X)):
			warnings.filterwarnings("ignore")
			if verbose > 0: print('validating...')
			model = SARIMAX(train_X, order=(n_lags, d, q), enforce_invertibility=False, enforce_stationarity=False)
			model_fit = model.fit(disp=verbose, iprint=verbose, maxiter=200, method='powell')
			output = model_fit.forecast()[0]
			y_hat_val.append(output)
			train_X = np.append(train_X, val_X[i])

		for i in range(test_size + 1):
			warnings.filterwarnings("ignore")
			if verbose > 0: print('testing...')
			model = SARIMAX(train_X, order=(n_lags, d, q), enforce_invertibility=False, enforce_stationarity=False)
			model_fit = model.fit(disp=verbose, iprint=verbose, maxiter=200, method='powell')
			output = model_fit.forecast()[0]
			y_hat.append(output)
			train_X = np.append(train_X, test_y[i])
		
		if(calc_val_error):
			tmp = np.zeros((len(y_hat_val), n_features))
			tmp[:, 0] = y_hat_val
			y_hat_val = scaler.inverse_transform(tmp)[:, 0]

			rmse_val = math.sqrt(mean_squared_error(val_X, y_hat_val))
		else:
			rmse_val = None
		
		if(calc_test_error):
			tmp = np.zeros((len(y_hat), n_features))
			tmp[:, 0] = y_hat
			y_hat = scaler.inverse_transform(tmp)[:, 0]

			rmse = math.sqrt(mean_squared_error(test_y, y_hat))
		else:
			rmse = None

		model = SARIMAX(train_X, order=(n_lags, d, q), enforce_invertibility=False, enforce_stationarity=False)
		if verbose > 0: print('predicting...')
		model_fit = model.fit(disp=verbose, iprint=verbose,)
		last = model_fit.forecast()[0]
		last = last.reshape(-1, 1)
		Xs = np.ones((last.shape[0], n_lags * n_features))
		inv_yhat = np.concatenate((last, Xs[:, -(n_features - n_series):]), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, 0:n_series]
	except (ValueError, LinAlgError):
		return 9e+10, 9e+10, None, None, None


	return rmse, rmse_val, test_y, y_hat, inv_yhat[-1]