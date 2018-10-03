import numpy as np
import pandas as pd
import math
import csv
import gc

import main
import modelos

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

#from keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector

from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
from hyperopt import STATUS_OK

from timeit import default_timer as timer


def normalize_data(values, scale=(0,1)):
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

def decompose_pca(X, n_pca, m_n):
	# Vector de medias
	mean_vector = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])

	# Matriz de dispersion
	scatter_matrix = np.zeros((X.shape[1], X.shape[1]))
	for i in range(X.shape[0]):
		scatter_matrix += (X[i,:].reshape(X.shape[1], 1) - mean_vector).dot((X[i,:].reshape(X.shape[1], 1) - mean_vector).T)

	eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

	# Prove de the equation: covariance matrix * eigenvector = eigenvalue*eigen vector
	for i in range(len(eig_val_sc)):
		eigv = eig_vec_sc[:,i].reshape(1, X.shape[1]).T
		np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv, decimal=6, err_msg='', verbose=True)

	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort(key=lambda x: x[0], reverse=True)

	# Create matrix W
	matrix_w = np.hstack([eig_pairs[i][1].reshape(X.shape[1],1) for i in range(n_pca)])

	sumas = np.array([np.sum(np.absolute(matrix_w[i,:])) for i in range(matrix_w.shape[0])])

	sumas_pairs = [(sumas[i], i) for i in range(sumas.shape[0])]
	sumas_pairs.sort(key=lambda x: x[0], reverse=False)

	indexes = [sumas_pairs[i][1] for i in range(m_n)]

	matrix_w[indexes] = matrix_w[indexes] * 0

	# Centered: that is why the minus mean vector
	transformed = (X - mean_vector).dot(matrix_w)

	return transformed

def transform_values(data, n_lags, n_series, dim, n_pca=0, m_n=0):
	global scaler
	train_size = 0.8
	n_features = data.shape[1]
	reframed = series_to_supervised(data, n_lags, n_series)

	# print(scaler.inverse_transform(reframed.values[-1,-50:].reshape(1, -1)))
	values = reframed.values # if n_lags = 1 then shape = (349, 100), if n_lags = 2 then shape = (348, 150)
	# n_examples for training set
	n_train = int(values.shape[0] * train_size)
	train = values[:n_train, :]
	test = values[n_train:, :]
	# observations for training, that is to say, series in the times (t-n_lags:t-1) taking t-1 because observations in time t is for testing
	n_obs = n_lags * n_features

	# for only testing y, y only contains target variable in different times
	# cols = [('var%d(t-%d)' % (j+1, i)) for j in range(n_features) for i in range(n_lags, 0, -1)]
	cols = ['var1(t)']
	cols += ['var1(t+%d)' % (i) for i in range(1, n_series)]
	y_o = reframed[cols].values
	train_o = y_o[:n_train]
	test_o = y_o[n_train:]

	if(-n_features + n_series == 0):
		train_X, train_y = train[:, :n_obs], train_o[:, -n_features:]
		test_X, test_y = test[:, :n_obs], test_o[:, -n_features:]
	else:
		#train_X, train_y = train[:, :n_obs], train_o[:, -n_series:]
		#test_X, test_y = test[:, :n_obs], test_o[:, -n_series:]
		train_X, train_y = train[:, :n_obs], train[:, -n_features:]
		test_X, test_y = test[:, :n_obs], test[:, -n_features:]
	
	# # PCA
	# train_X = train_X.reshape((train_X.shape[0]* n_lags, n_features))
	# test_X = test_X.reshape((test_X.shape[0]* n_lags, n_features))

	# X = np.concatenate((train_X, test_X), axis=0)

	# # My implementation
	# X = decompose_pca(X, n_pca, m_n)
	# # # scikit learn implementation
	# # pca = PCA(n_components=n_pca)
	# # X = pca.fit_transform(X)

	# train_X, test_X = X[:n_train*n_lags], X[n_train*n_lags:]

	# train_X = train_X.reshape((int(train_X.shape[0]/n_lags), n_lags, n_pca))
	# test_X = test_X.reshape((int(test_X.shape[0]/n_lags), n_lags, n_pca))


	# implementation without PCA
	# reshape input to be 3D [n_examples, timesteps, features]
	if(dim):
		train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
		test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
		last_values = np.append(test_X[-1,1:n_lags,:], [test[-1,-n_features:]], axis=0)
		# last_values = np.insert(test_X[-10:,1:n_lags,:], n_lags-1, test[-10:,-n_features:], axis=1)
	else:
		last_values = np.append(test_X[-1, n_features:].reshape(1,-1), [test[-1,-n_features:]], axis=1)
	return train_X, test_X, train_y, test_y, last_values

# def train_model(train_X, test_X, train_y, test_y, n_series, n_a, n_epochs, batch_size, i_model):
def train_model(train_X, test_X, train_y, test_y, n_series, params, i_model, n_features, n_lags, scaler, last_values):
	if(i_model == 0):
		return modelos.model_lstm(train_X, test_X, train_y, test_y, n_series, params['n_epochs'], params['batch_size'], params['lr'], params['n_hidden'], n_features, n_lags, scaler, last_values)
	elif(i_model == 1):
		return modelos.model_random_forest(train_X, test_X, train_y, test_y, n_series, params['n_estimators'], params['max_features'], params['min_samples'], n_features, n_lags, scaler, last_values)
	elif(i_model == 2):
		return modelos.model_ada_boost(train_X, test_X, train_y, test_y, n_series, params['n_estimators'], params['lr'], n_features, n_lags, scaler, last_values)
	elif(i_model == 3):
		return modelos.model_svm(train_X, test_X, train_y, test_y, n_series, n_features, n_lags, scaler, last_values)
	elif(i_model == 4):
		return modelos.model_arima(train_X, test_X, train_y, test_y, n_series, params['d'], params['q'], n_features, n_lags, scaler, last_values)

def plot_data(data, labels, title):
	plt.figure()
	for i in range(len(data)):
		plt.plot(data[i], label=labels[i])
	plt.suptitle(title, fontsize=16)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
	plt.show()

def plot_data_lagged(data, labels, title):
	plt.figure()
	plt.plot(data[0][0, :], label=labels[0])
	for i in range(len(data[1])):
		print(i)
		padding = [None for j in range(i*len(data[1][0]))]
		print(padding)
		print(data[1][i])
		print(padding + data[1][i]) # esto debe funcionar para graficar los datos con lag y mirar tendencia como en el paper que mando daniel en la funcion plot_results_multiple
		plt.plot(padding + data[1][i], label=labels[1] + str(i+1))
	plt.suptitle(title, fontsize=16)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
	plt.show()

# for bayes optimization
def objective(params):
	
	# Keep track of evals
	global ITERATION, values, scaler, n_series, train_size, i_model
	
	ITERATION += 1
	out_file = 'gbm_trials.csv'
	print(ITERATION, params)

	if(i_model == 0):
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_epochs', 'batch_size', 'n_hidden']:
			params[parameter_name] = int(params[parameter_name])

		for parameter_name in ['lr']:
			params[parameter_name] = float(params[parameter_name])

		start = timer()
		train_X, test_X, train_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 1)
		rmse, _, _, _ = train_model(train_X, test_X, train_y, test_y, n_series, {'n_epochs':params['n_epochs'], 'batch_size':params['batch_size'], 'lr':params['lr'], 'n_hidden':params['n_hidden']}, i_model, n_features, params['n_lags'], scaler, last_values)
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

def bayes_optimization(i_model):
	global ITERATION
	ITERATION = 0

	if(i_model == 0):
		# space
		space = {'n_lags': hp.quniform('n_lags', 1, 50, 1),
				'n_epochs': hp.quniform('n_epochs', 10, 200, 1),
				'batch_size': hp.quniform('batch_size', 5, 100, 1),
				'n_hidden': hp.quniform('n_hidden', 5, 300, 1),
				'lr': hp.uniform('lr', 0.0001, 1.0)}
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
	best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(np.random.randint(100)))

	# store best results
	of_connection = open('bests.txt', 'a')
	writer = csv.writer(of_connection)
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	if(i_model == 0):
		writer.writerow([bayes_trials_results[0]['loss'], bayes_trials_results[0]['params']['n_lags'], bayes_trials_results[0]['params']['n_hidden'], bayes_trials_results[0]['params']['n_epochs'], bayes_trials_results[0]['params']['batch_size'], MAX_EVALS])
	elif(i_model == 1):
		writer.writerow([bayes_trials_results[0]['loss'], best['n_lags'], best['n_estimators'], best['max_features'], best['min_samples'], MAX_EVALS])
	elif(i_model == 2):
		writer.writerow([bayes_trials_results[0]['loss'], best['n_lags'], best['n_estimators'], best['lr'], MAX_EVALS])
	elif(i_model == 3):
		writer.writerow([bayes_trials_results[0]['loss'], best['n_lags'], MAX_EVALS])
	of_connection.close()

	return best, bayes_trials_results


def predictor(data, id_model, tune, select, original):
	global values, scaler, n_features, MAX_EVALS, n_series, train_size, i_model
	
	if(select):
		# feature selection
		import feature_selection
		# feature_selection.select_features_sa(pd.DataFrame(data))
		df = pd.read_csv('data/forecast-competition-complete.csv', index_col=0, header=0)
		feature_selection.select_features_stepwise_forward(df, 5)
		# feature_selection.select_features_ga(pd.DataFrame(data))
	if(not original):
		# df = pd.read_csv('data/forecast-competition-complete_selected.csv', index_col=0)
		df = pd.read_csv('data/forecast-competition-complete_selected_manually.csv', index_col=0)
		data = df.values
	# values = data

	values, scaler = normalize_data(data)
	MAX_EVALS = 200
	n_series = 1
	train_size = 0.8

	n_features = values.shape[1]
	i_model = id_model

	if(tune):
		if(i_model == 0):
			# Parameters
			best, results = bayes_optimization(i_model)
			n_lags, n_epochs, batch_size, lr, n_hidden = int(best['n_lags']), int(best['n_epochs']), int(best['batch_size']), float(best['lr']), int(best['n_hidden'])
			
			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 1)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_epochs':n_epochs, 'batch_size':batch_size, 'lr':lr, 'n_hidden':n_hidden}, i_model, n_features, n_lags, scaler, last_values)
			#plot_data([history.history['loss'], history.history['val_loss']], ['loss', 'val_loss'], 'Loss plot')

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			# print(y_hat[-1][0])
			return last
		elif(i_model == 1):
			best, results = bayes_optimization(i_model)
			n_lags, n_estimators, max_features, min_samples = int(best['n_lags']), int(best['n_estimators']), int(best['max_features']), int(best['min_samples'])

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'max_features':max_features, 'min_samples':min_samples}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 2):
			best, results = bayes_optimization(i_model)
			n_lags, n_estimators, lr = int(best['n_lags']), int(best['n_estimators']), best['lr']

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'lr':lr}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 3):
			best, results = bayes_optimization(i_model)
			n_lags = int(best['n_lags'])

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last

	else:
		if(i_model == 0):
			# without selecting features
			# n_lags, n_a, n_epochs, batch_size = 3,163,58,35 # 3,250,151,42,5,34#3,194,51,44,11,27#6,226,84,100,4,5#2, 250, 25, 50, 15 #1,12,130,225, 31 #7, 16, 94, 93, 49
			# with feature selection
			# batch_size, lr, n_a, n_epochs, n_hidden, n_lags = 75, 0.0001, 274, 191, 50, 31
			# batch_size, lr, n_epochs, n_hidden, n_lags = 75, 0.0001, 91, 50, 10
			# for model with perd to next time and only target variable:
			# batch_size, lr, n_epochs, n_hidden, n_lags = 67, 0.9974425873058935, 19, 249, 6
			
			# batch_size, lr, n_epochs, n_hidden, n_lags = 35, 0.001, 58, 163, 3
			batch_size, lr, n_epochs, n_hidden, n_lags = 50, 0.4799370248396754, 106, 25, 3

			############# ************** con 0.001 de lr funciona bien con 0.0001 tambien pero con valores diferentes hay vanishing gradients problem
			##### una intuicion es que con menos lags se multiplican menos los gradietnes y se vuelven menos pequeño por que un numero pequeño por otro peuqueño se vuelve aun mas pequeño
			###### los parametros de inicializar los Ws del RNN estan guardados, intentar tambien guardar unos buenos para las capas lineales

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 1)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_epochs':n_epochs, 'batch_size':batch_size, 'lr':lr, 'n_hidden':n_hidden}, i_model, n_features, n_lags, scaler, last_values)
			#plot_data([history.history['loss'], history.history['val_loss']], ['loss', 'val_loss'], 'Loss plot')
			print('rmse: %s ' % rmse)
			plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')
			# plot_data_lagged([y, y_hat], ['y', 'y_hat'], 'Test plot')

			#for i in range(10):
			#	plot_data([y[i], y_hat[i]], ['y', 'y_hat'], 'Test plot')

			# return y_hat[-1]
			# print(y_hat[-1][0])
			return last[0][0]
		elif(i_model == 1):
			# n_lags, n_estimators, max_features, min_samples = 4, 500, 14, 1
			n_lags, n_estimators, max_features, min_samples = 4, 762, 18, 3 # for selected features

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'max_features':max_features, 'min_samples':min_samples}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 2):
			n_lags, n_estimators, lr = 4, 808, 0.33209425848535884

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {'n_estimators':n_estimators, 'lr':lr}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 3):
			n_lags = 4

			train_X, test_X, train_y, test_y, last_values = transform_values(values, n_lags, n_series, 0)
			rmse, y, y_hat, last = train_model(train_X, test_X, train_y, test_y, n_series, {}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last
		elif(i_model == 4):
			n_lags, d, q = 5, 2, 1
			wall = int(len(values)*0.8)
			train, test, last_values = values[:wall, 0], values[wall:-1,0], values[-1,0]
			rmse, y, y_hat, last = train_model(train, [], [], test, n_series, {'d':d, 'q':q}, i_model, n_features, n_lags, scaler, last_values)

			print('rmse: %s ' % rmse)
			#plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

			return last





if __name__ == '__main__':
	main.main()