import numpy as np
import pandas as pd
import math
import csv

import main
import modelos

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
from hyperopt import STATUS_OK

from timeit import default_timer as timer


def normalize_data(values, scale=(-1,1)):
	# Be sure all values are numbers
	values = values.astype('float32')
	# scale the data
	scaler = MinMaxScaler(feature_range=scale)
	scaled = scaler.fit_transform(values)
	return scaled, scaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# For no slidding window approach
def split_data(data):
	n_train = int(len(data)*0.6)
	n_val = int(len(data)*0.2)
	ys = data[1:, 0]

	X_train, y_train = data[:n_train, :], data[1:n_train+1]
	X_val, y_val = data[n_train:n_train+n_val, :], data[n_train+1:n_train+n_val+1]
	X_test, y_test = data[n_train+n_val:-1, :], data[n_train+n_val+1:]

	last_values = data[-1]

	return X_train, X_val, X_test, y_train, y_val, y_test, last_values

def transform_values(data, n_lags, n_series, dim):
	train_size = 0.6
	val_size =0.2
	test_size = 0.2
	n_features = data.shape[1]
	reframed = series_to_supervised(data, n_lags, n_series)

	values = reframed.values # if n_lags = 1 then shape = (349, 100), if n_lags = 2 then shape = (348, 150)
	# n_examples for training set
	n_train = int(values.shape[0] * train_size)
	n_val = int(values.shape[0] * val_size)
	train = values[:n_train, :]
	val = values[n_train:n_train + n_val, :]
	test = values[n_train + n_val:, :]
	# observations for training, that is to say, series in the times (t-n_lags:t-1) taking t-1 because observations in time t is for testing
	n_obs = n_lags * n_features

	# for only testing y, y only contains target variable in different times
	# cols = [('var%d(t-%d)' % (j+1, i)) for j in range(n_features) for i in range(n_lags, 0, -1)]
	cols = ['var1(t)']
	cols += ['var1(t+%d)' % (i) for i in range(1, n_series)]
	y_o = reframed[cols].values
	train_o = y_o[:n_train]
	val_o = y_o[n_train:n_train + n_val]
	test_o = y_o[n_train + n_val:]

	train_X, train_y = train[:, :n_obs], train_o[:, -n_series:]
	val_X, val_y = val[:, :n_obs], val_o[:, -n_series:]
	test_X, test_y = test[:, :n_obs], test_o[:, -n_series:]
	#train_X, train_y = train[:, :n_obs], train[:, n_obs:n_obs+n_features]
	#test_X, test_y = test[:, :n_obs], test[:, n_obs:n_obs+n_features]

	# reshape train data to be 3D [n_examples, n_lags, features]
	if(dim):
		train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
		val_X = val_X.reshape((val_X.shape[0], n_lags, n_features))
		test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
		last_values = np.append(test_X[-1,1:n_lags,:], [test[-1,-n_features:]], axis=0)
		# last_values = np.insert(test_X[-10:,1:n_lags,:], n_lags-1, test[-10:,-n_features:], axis=1)
	else:
		# print('falta implementar validacion para esta no dim en transform values')
		# raise Exception('falta implementar validacion para no dim en transform values')
		last_values = np.append(test_X[-1, n_features:].reshape(1,-1), [test[-1,-n_features:]], axis=1)
		return train_X, test_X, train_y, test_y, last_values
	return train_X, val_X, test_X, train_y, val_y, test_y, last_values

# def train_model(train_X, test_X, train_y, test_y, n_series, n_a, n_epochs, batch_size, i_model):
def train_model(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params, i_model, n_features, n_lags, scaler, last_values):
	if(i_model == 0):
		return modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params['n_epochs'], params['batch_size'], params['n_hidden'], n_features, n_lags, scaler, last_values)
	elif(i_model == 1):
		return modelos.model_random_forest(train_X, test_X, train_y, test_y, n_series, params['n_estimators'], params['max_features'], params['min_samples'], n_features, n_lags, scaler, last_values)
	elif(i_model == 2):
		return modelos.model_ada_boost(train_X, test_X, train_y, test_y, n_series, params['n_estimators'], params['lr'], n_features, n_lags, scaler, last_values)
	elif(i_model == 3):
		return modelos.model_svm(train_X, test_X, train_y, test_y, n_series, n_features, n_lags, scaler, last_values)
	elif(i_model == 4):
		return modelos.model_arima(train_X, test_X, train_y, test_y, n_series, params['d'], params['q'], n_features, n_lags, scaler, last_values)
	elif(i_model == 5):
		return modelos.model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params['n_epochs'], params['n_hidden'], params['lr'], n_features, n_lags, scaler, last_values)

def plot_data(data, labels, title):
	plt.figure()
	for i in range(len(data)):
		plt.plot(data[i], label=labels[i])
	plt.suptitle(title, fontsize=16)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
	plt.show()

def plot_data_lagged(data, labels, title):
	plt.figure()
	plt.plot(data[0], label=labels[0])
	for i in range(len(data[1])):
		padding = [None for _ in range(i)]
		plt.plot(padding + list(data[1][i]), label=labels[1] + str(i+1))
	plt.suptitle(title, fontsize=16)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
	plt.show()

def plot_data_lagged_blocks(data, labels, title):
	plt.figure()
	plt.plot(data[0], label=labels[0])
	for i in range(0, len(data[1]), len(data[1][0])):
		padding = [None for _ in range(i)]
		plt.plot(padding + list(data[1][i]), label=labels[1] + str(i+1))
	plt.suptitle(title, fontsize=16)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
	plt.show()

# for bayes optimization
def objective(params, values, scaler, n_series, i_model):
	
	# Keep track of evals
	global ITERATION
	
	ITERATION += 1
	out_file = 'gbm_trials.csv'
	print(ITERATION, params)

	if(i_model == 0):
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_epochs', 'batch_size', 'n_hidden']:
			params[parameter_name] = int(params[parameter_name])

		#for parameter_name in ['lr']:
		#	params[parameter_name] = float(params[parameter_name])

		start = timer()
		train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 1)
		rmse, _, _, _ = train_model(train_X, val_X, test_X, train_y, val_y, test_y, n_series, {'n_epochs':params['n_epochs'], 'batch_size':params['batch_size'], 'n_hidden':params['n_hidden']}, i_model, n_features, params['n_lags'], scaler, last_values)
		run_time = timer() - start
		print('rmse: ', rmse)
		print('time: ', run_time, end='\n\n')
	elif(i_model == 1):
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_estimators', 'max_features', 'min_samples']:
			params[parameter_name] = int(params[parameter_name])


		start = timer()
		train_X, test_X, train_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 0)
		rmse, _, _, _ = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':params['n_estimators'], 'max_features':params['max_features'], 'min_samples':params['min_samples']}, i_model, n_features, params['n_lags'], scaler, last_values)
		run_time = timer() - start
		print('rmse: ', rmse)
		print('time: ', run_time, end='\n\n')
	elif(i_model == 2):
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_estimators']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, test_X, train_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 0)
		rmse, _, _, _ = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':params['n_estimators'], 'lr':params['lr']}, i_model, n_features, params['n_lags'], scaler, last_values)
		run_time = timer() - start
		print('rmse: ', rmse)
		print('time: ', run_time, end='\n\n')
	elif(i_model == 3):
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, test_X, train_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 0)
		rmse, _, _, _ = train_model(train_X, test_X, train_y, test_y, n_series, {}, i_model, n_features, params['n_lags'], scaler, last_values)
		run_time = timer() - start
		print('rmse: ', rmse)
		print('time: ', run_time, end='\n\n')

	# Write to the csv file ('a' means append)
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([rmse, params, ITERATION, run_time])
	of_connection.close()
	del start, of_connection, train_X, test_X, train_y, test_y
	gc.collect()

	# Dictionary with information for evaluation
	return {'loss': rmse, 'params': params, 'iteration': ITERATION,
			'train_time': run_time, 'status': STATUS_OK}

def bayes_optimization(i_model, MAX_EVALS, values, scaler, n_features, n_series):
	global ITERATION
	ITERATION = 0

	if(i_model == 0):
		# space
		# big
		# space = {'n_lags': hp.quniform('n_lags', 1, 50, 1),
		# 		'n_epochs': hp.quniform('n_epochs', 10, 200, 1),
		# 		'batch_size': hp.quniform('batch_size', 5, 100, 1),
		# 		'n_hidden': hp.quniform('n_hidden', 5, 300, 1)}
		# space = {'n_lags': hp.quniform('n_lags', 20, 30, 1),
		# 		'n_epochs': hp.quniform('n_epochs', 10, 200, 1),
		# 		'batch_size': hp.quniform('batch_size', 5, 100, 1),
		# 		'n_hidden': hp.quniform('n_hidden', 155, 170, 1)}
		space = {'n_lags': hp.quniform('n_lags', 40, 80, 1),
				'n_epochs': hp.quniform('n_epochs', 2, 100, 1),
				'batch_size': hp.quniform('batch_size', 15, 70, 1),
				'n_hidden': hp.quniform('n_hidden', 200, 400, 1)}
	elif(i_model == 1):
		# space = {'n_lags': hp.quniform('n_lags', 1, 50, 1),
		# 		'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
		# 		'max_features': hp.quniform('max_features', 1, 50, 1),
		# 		'min_samples': hp.quniform('min_samples', 1, 20, 1)}
		space = {'n_lags': hp.quniform('n_lags', 1, 50, 1),
		 		'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
		 		'max_features': hp.quniform('max_features', 1, 23, 1),
		 		'min_samples': hp.quniform('min_samples', 1, 20, 1)}
	elif(i_model == 2):
		space = {'n_lags': hp.quniform('n_lags', 1, 50, 1),
				'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
				'lr': hp.uniform('lr', 0.00001, 1.0)}
	elif(i_model == 3):
		space = {'n_lags': hp.quniform('n_lags', 1, 50, 1)}

	# Keep track of results
	bayes_trials = Trials()

	# File to save first results
	out_file = 'gbm_trials.csv'
	of_connection = open(out_file, 'w')
	writer = csv.writer(of_connection)

	# Write the headers to the file
	writer.writerow(['rmse', 'params', 'iteration', 'train_time'])
	of_connection.close()

	# Run optimization
	best = fmin(fn = lambda x: objective(x, values, scaler, n_series, i_model), space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(np.random.randint(100)))

	# store best results
	of_connection = open('bests.txt', 'a')
	writer = csv.writer(of_connection)
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	if(i_model == 0):
		writer.writerow([bayes_trials_results[0]['loss'], bayes_trials_results[0]['params']['batch_size'], bayes_trials_results[0]['params']['n_epochs'], bayes_trials_results[0]['params']['n_hidden'], bayes_trials_results[0]['params']['n_lags'], MAX_EVALS])
	elif(i_model == 1):
		writer.writerow([bayes_trials_results[0]['loss'], best['n_lags'], best['n_estimators'], best['max_features'], best['min_samples'], MAX_EVALS])
	elif(i_model == 2):
		writer.writerow([bayes_trials_results[0]['loss'], best['n_lags'], best['n_estimators'], best['lr'], MAX_EVALS])
	elif(i_model == 3):
		writer.writerow([bayes_trials_results[0]['loss'], best['n_lags'], MAX_EVALS])
	of_connection.close()

	return best, bayes_trials_results


def predictor(data, id_model, tune, select, original, time_steps, max_vars):
	global values, scaler, n_features, MAX_EVALS, n_series, i_model
	
	if(select):
		# feature selection
		import feature_selection
		# feature_selection.select_features_sa(pd.DataFrame(data), max_vars)
		feature_selection.select_features_ga(pd.DataFrame(data), max_vars)
		# df = pd.read_csv('data/forecast-competition-complete.csv', index_col=0, header=0)
		# feature_selection.select_features_stepwise_forward(df, max_vars)
	if(not original or select):
		df = pd.read_csv('data/forecast-competition-complete_selected.csv', index_col=0)
		# df = pd.read_csv('data/forecast-competition-complete_selected_manually.csv', index_col=0)
		data = df.values
	# values = data

	values, scaler = normalize_data(data)
	MAX_EVALS = 200
	n_series = time_steps

	n_features = values.shape[1]
	i_model = id_model

	if(tune):
		if(i_model == 0):
			# Parameters
			best, results = bayes_optimization(i_model, MAX_EVALS, values, scaler, n_features, n_series)
			n_lags, n_epochs, batch_size, n_hidden = int(best['n_lags']), int(best['n_epochs']), int(best['batch_size']), int(best['n_hidden'])
			
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, n_lags, n_series, 1)
			rmse, y, y_hat, last = train_model(train_X, val_X, test_X, train_y, val_y, test_y, n_series, {'n_epochs':n_epochs, 'batch_size':batch_size, 'n_hidden':n_hidden}, i_model, n_features, n_lags, scaler, last_values)
			#plot_data([history.history['loss'], history.history['val_loss']], ['loss', 'val_loss'], 'Loss plot')

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			# print(y_hat[-1][0])
			return last
		elif(i_model == 1 and n_series == 1):
			best, results = bayes_optimization(i_model)
			n_lags, n_estimators, max_features, min_samples = int(best['n_lags']), int(best['n_estimators']), int(best['max_features']), int(best['min_samples'])

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'max_features':max_features, 'min_samples':min_samples}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 2 and n_series == 1):
			best, results = bayes_optimization(i_model)
			n_lags, n_estimators, lr = int(best['n_lags']), int(best['n_estimators']), best['lr']

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'lr':lr}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 3 and n_series == 1):
			best, results = bayes_optimization(i_model)
			n_lags = int(best['n_lags'])

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last

	else:
		if(i_model == 0):			
			if(original and not select):
				# batch_size, lr, n_epochs, n_hidden, n_lags = 35, 0.001, 58, 163, 3
				# approach for max 8 lags
				# batch_size, lr, n_epochs, n_hidden, n_lags = 50, 0.4799370248396754, 106, 25, 3
				# approach for max 50 lags
				batch_size, lr, n_epochs, n_hidden, n_lags = 52, 0.4799370248396754, 33, 159, 28
			else:
				batch_size, n_epochs, n_hidden, n_lags = 15, 91, 24, 2
				# batch_size, n_epochs, n_hidden, n_lags = 45, 19, 50, 32

			# print(batch_size, n_epochs, n_hidden, n_lags)
			# train_X, val_X, test_X, train_y, val_y, test_y, last_values = split_data(data)
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, n_lags, n_series, 1)
			rmse, y, y_hat, last = train_model(train_X, val_X, test_X, train_y, val_y, test_y, n_series, {'n_epochs':n_epochs, 'batch_size':batch_size, 'n_hidden':n_hidden}, i_model, n_features, n_lags, scaler, last_values)
			#plot_data([history.history['loss'], history.history['val_loss']], ['loss', 'val_loss'], 'Loss plot')
			print('rmse: %s ' % rmse)
			# plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')
			#plot_data_lagged([test_y[:, 0].ravel(), y_hat[:, :, 0].squeeze()], ['y', 'y_hat'], 'Test plot')
			#plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat[:, :, 0].squeeze()], ['y', 'y_hat'], 'Test plot')
			# plot_data_lagged([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
			# plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
			# plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Validation plot')

			# # print test predictions vs observations
			# mini = 0
			# maxi = 64
			# step = 8
			# n_rows = 2
			# n_cols = step/n_rows
			# for i in range(mini, maxi, step):
			# 	# plot_data([y[i, :, 0], y_hat[i, :, 0]], ['y', 'y_hat'], 'Test plot')
			# 	for j in range(step):
			# 		plt.subplot(n_rows,n_cols,j+1)
			# 		plt.plot(y[i+j, :, 0])
			# 		plt.plot(y_hat[i+j, :, 0])
			# 	plt.show()

			return last
		elif(i_model == 1 and n_series == 1):
			# n_lags, n_estimators, max_features, min_samples = 4, 500, 14, 1
			n_lags, n_estimators, max_features, min_samples = 4, 762, 18, 3 # for selected features

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'max_features':max_features, 'min_samples':min_samples}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 2 and n_series == 1):
			n_lags, n_estimators, lr = 4, 808, 0.33209425848535884

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'lr':lr}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 3 and n_series == 1):
			n_lags = 4

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 4 and n_series == 1):
			n_lags, d, q = 5, 2, 1
			wall = int(len(values)*0.8)
			train, test, last_values = values[:wall, 0], values[wall:-1,0], values[-1,0]
			rmse, y, y_hat, last = train_model(train, [], [], test, n_series, {'d':d, 'q':q}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 5):
			lr, n_epochs, n_hidden = 0.001, 300, 159
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = split_data(data)
			rmse, y, y_hat, last = train_model(train_X, val_X, test_X, train_y, val_y, test_y, n_series, {'n_epochs':n_epochs, 'n_hidden':n_hidden, 'lr':lr}, i_model, n_features, -1, scaler, last_values)
			return last
		else:
			raise Exception('parameters combination is not in the valid options.')





if __name__ == '__main__':
	main.main()