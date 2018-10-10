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
	rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat)) / np.max(inv_y) * 100
	return inv_y, inv_yhat, rmse

def predict_last(n_series, n_features, n_lags, X, scaler, model, dim):
	if(dim):
		X = np.expand_dims(X, axis=0)
	yhat = model.predict(X)
	if(not dim):
		yhat = yhat.reshape(-1, 1)
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
def model_lstm(train_X, test_X, train_y, test_y, val_X, val_y, n_series, n_epochs, batch_size, n_hidden, n_features, n_lags, scaler, last_values):
	from keras.layers import Dense, Activation, Dropout, LSTM
	from keras.models import Sequential
	from keras.optimizers import Adam

	drop_p = 0.05
	n_out = n_series

	model = Sequential()
	model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
	model.add(Dense(n_out, input_shape=(n_hidden,)))

	opt = Adam(lr=0.001, decay=0.0)
	model.compile(loss='mse', optimizer=opt)
	model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=0, shuffle=False)

	
	y = test_y
	y_hat = model.predict(test_X)
	y_hat_val = model.predict(val_X)

	# predict last
	last = model.predict(np.expand_dims(last_values, axis=0))

	# for test
	rmses = []
	rmse = 0
	weigth = 1.0
	step = 0.1
	for i in range(n_out):
		rmses.append(math.sqrt(mean_squared_error(y[:, i], y_hat[:, i])))
		rmse += rmses[-1]*weigth
		weigth -= step

	# for validation
	rmses_val = []
	rmse_val = 0
	weigth = 1.0
	step = 0.1
	for i in range(n_out):
		rmses_val.append(math.sqrt(mean_squared_error(val_y[:, i], y_hat_val[:, i])))
		rmse_val += rmses_val[-1]*weigth
		weigth -= step

	# transform last values
	tmp = np.zeros((last.shape[1], n_features))
	tmp[:, 0] = last
	last = scaler.inverse_transform(tmp)[:, 0]

	return rmse_val, val_y, y_hat_val, last


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

	rmse = math.sqrt(mean_squared_error(test_y, y_hat))
	model = SARIMAX(train_X, order=(n_lags, d, q))
	model_fit = model.fit(disp=0)
	last = model_fit.forecast()[0]
	last = last.reshape(-1, 1)
	Xs = np.ones((last.shape[0], n_lags * n_features))
	inv_yhat = np.concatenate((last, Xs[:, -(n_features - n_series):]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)

	inv_yhat = inv_yhat[:,0:n_series]

	return rmse, test_y, y_hat, inv_yhat[-1]