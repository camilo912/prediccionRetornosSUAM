import numpy as np
import pandas as pd
import math
import utils

import modelos

from timeit import default_timer as timer


class Predictor():
	def __init__(self, id_model, original):
		self.id_model = id_model
		self.original = original
		if(self.id_model == 0):
			if(self.original):
				self.batch_size, self.lr, self.n_epochs, self.n_hidden, self.n_lags = 52, 0.4799370248396754, 33, 159, 28 
			else:
				self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 30, 7, 288, 65 # 15, 91, 24, 2
		
		elif(self.id_model == 1):
			self.n_lags, self.n_estimators, self.max_features, self.min_samples = 4, 762, 18, 3 # for selected features
		
		elif(self.id_model == 2):
			self.n_lags, self.n_estimators, self.lr, self.max_depth = 4, 808, 0.33209425848535884, 3
		
		elif(self.id_model == 3):
			self.n_lags = 4
		
		elif(self.id_model == 4):
			self.n_lags, self.d, self.q = 3, 0, 6 # best
			# self.n_lags, self.d, self.q = 12, 0 , 1
			# self.n_lags, self.d, self.q = 5, 2, 1
			# self.n_lags, self.d, self.q = 11, 1, 2
			# self.n_lags, self.d, self.q = 1, 0, 7
			# self.n_lags, self.d, self.q = 2, 2, 11 # too slow

		
		elif(self.id_model == 5):
			if(self.original):
				self.lr, self.n_epochs = 0.014540077212290003, 37
			else:
				self.lr, self.n_epochs = 0.2050774898644015, 22


	def predict(self, data, original_cols, tune, select, time_steps, max_vars, verbosity, parameters_file_name=None, max_evals=100):
		if(select):
			self.original = 0
			# feature selection
			import feature_selection
			# feature_selection.select_features_sa(pd.DataFrame(data), max_vars)
			feature_selection.select_features_ga(pd.DataFrame(data), max_vars, self.original_cols)
			# df = pd.read_csv('data/forecast-competition-complete.csv', index_col=0, header=0)
			# feature_selection.select_features_stepwise_forward(df, max_vars)
		
		if(not self.original):
			df = pd.read_csv('data/forecast-competition-complete_selected.csv', index_col=0)
			# df = pd.read_csv('data/forecast-competition-complete_selected_manually.csv', index_col=0)
			data = df.values

		self.MAX_EVALS = max_evals
		values, scaler = utils.normalize_data(data)
		n_series = time_steps
		calc_val_error = False
		calc_test_error = True

		n_features = values.shape[1]

		if(tune):
			best, results = utils.bayes_optimization(self.id_model, self.MAX_EVALS, values, scaler, n_features, n_series, self.original, verbosity)

			if(self.id_model == 0):
				# Parameters
				self.n_lags, self.n_epochs, self.batch_size, self.n_hidden = int(best['n_lags']), int(best['n_epochs']), int(best['batch_size']), int(best['n_hidden'])
				f = open('optimized_lstm_%dtimesteps.pars' % time_steps, 'w')
				f.write('%d, %d, %d, %d\n' % (self.n_lags, self.n_epochs, self.batch_size, self.n_hidden))
				f.close()
				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 1)
				
				rmse, _, y, y_hat, last = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.batch_size, self.n_hidden, n_features, self.n_lags, scaler, 
																	last_values, calc_val_error, calc_test_error, verbosity)
				
				print('rmse: %s ' % rmse)
				
				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last
			
			elif(self.id_model == 1 and n_series == 1):
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = int(best['n_lags']), int(best['n_estimators']), int(best['max_features']), int(best['min_samples'])
				f = open('optimized_randomForest_%dtimesteps.pars' % time_steps, 'w')
				f.write('%d, %d, %d, %d\n' % (self.n_lags, self.n_estimators, self.max_features, self.min_samples))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				rmse, _, y, y_hat, last = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.max_features, self.min_samples, n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity)
				
				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 2 and n_series == 1):
				self.n_lags, self.n_estimators, self.lr, self.max_depth = int(best['n_lags']), int(best['n_estimators']), best['lr'], best['max_depth']
				f = open('optimized_adaBoost_%dtimesteps.pars' % time_steps, 'w')
				f.write('%d, %d, %f\n' % (self.n_lags, self.n_estimators, self.lr))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				rmse, _, y, y_hat, last = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.lr, self.max_depth, n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity)

				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 3 and n_series == 1):
				self.n_lags = int(best['n_lags'])
				f = open('optimized_SVM_%dtimesteps.pars' % time_steps, 'w')
				f.write('%d\n' % (self.n_lags))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				rmse, _, y, y_hat, last = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity)
				
				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 4 and n_series == 1):
				self.n_lags, self.d, self.q = int(best['n_lags']), int(best['d']), int(best['q'])
				f = open('optimized_arima_%dtimesteps.pars' % time_steps, 'w')
				f.write('%d, %d, %d\n' % (self.n_lags, self.d, self.q))
				f.close()

				wall = int(len(values)*0.6)
				wall_val= int(len(values)*0.2)
				train, val, test, last_values = values[:wall, 0], values[wall:wall+wall_val,0], values[wall+wall_val:-1,0], values[-1,0]
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_arima(train, val, [], [], [], test, n_series, self.d, self.q, n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 5):
				self.lr, self.n_epochs = best['lr'], int(best['n_epochs'])
				f = open('optimized_lstmNoSW_%dtimesteps.pars' % time_steps, 'w')
				f.write('%f, %d\n' % (self.lr, self.n_epochs))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.split_data(data)
				rmse, _, y, y_hat, last = modelos.model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.lr, n_features, -1, scaler, last_values, 
																				calc_val_error, calc_test_error, verbosity)

				print('rmse: %s ' % rmse)
				
				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last
			
			else:
				raise Exception('hyperparameters combination is not in the valid options.')
		
		else:
			if(self.id_model == 0):			
				if(parameters_file_name):
					try:
						f = open(parameters_file_name, 'r')
					except FileNotFoundError:
						raise Exception('No existe el archivo: %s' % parameters_file_name)
					lines = f.readlines()
					if(len(lines) > 1):
						raise Exception('File with parameters can\'t have more than 1 line ')
					readed_parameters = lines[0].strip().split(', ')
					self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = int(readed_parameters[0]), int(readed_parameters[1]), int(readed_parameters[2]), int(readed_parameters[3])

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 1)
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.batch_size, self.n_hidden, n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)
				
				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last
			
			elif(self.id_model == 1 and n_series == 1):
				if(parameters_file_name):
					try:
						f = open(parameters_file_name, 'r')
					except FileNotFoundError:
						raise Exception('No existe el archivo: %s' % parameters_file_name)
					lines = f.readlines()
					if(len(lines) > 1):
						raise Exception('File with parameters can\'t have more than 1 line ')
					readed_parameters = lines[0].strip().split(', ')

					self.n_lags, self.n_estimators, self.max_features, self.min_samples = int(readed_parameters[0]), int(readed_parameters[1]), int(readed_parameters[2]), int(readed_parameters[3])

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.max_features, self.min_samples, n_features, self.n_lags, 
																			scaler, last_values, calc_val_error, calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 2 and n_series == 1):
				if(parameters_file_name):
					try:
						f = open(parameters_file_name, 'r')
					except FileNotFoundError:
						raise Exception('No existe el archivo: %s' % parameters_file_name)
					lines = f.readlines()
					if(len(lines) > 1):
						raise Exception('File with parameters can\'t have more than 1 line ')
					readed_parameters = lines[0].strip().split(', ')

					self.n_lags, self.n_estimators, self.lr = int(readed_parameters[0]), int(readed_parameters[1]), float(readed_parameters[2])

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.lr, self.max_depth, n_features, self.n_lags, scaler, last_values, 
																		calc_val_error, calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 3 and n_series == 1):
				if(parameters_file_name):
					try:
						f = open(parameters_file_name, 'r')
					except FileNotFoundError:
						raise Exception('No existe el archivo: %s' % parameters_file_name)
					lines = f.readlines()
					if(len(lines) > 1):
						raise Exception('File with parameters can\'t have more than 1 line ')
					readed_parameters = lines[0].strip().split(', ')
					self.n_lags = int(readed_parameters[0])

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, self.n_lags, scaler, last_values, calc_val_error, 
																	calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 4 and n_series == 1):
				if(parameters_file_name):
					try:
						f = open(parameters_file_name, 'r')
					except FileNotFoundError:
						raise Exception('No existe el archivo: %s' % parameters_file_name)
					lines = f.readlines()
					if(len(lines) > 1):
						raise Exception('File with parameters can\'t have more than 1 line ')
					readed_parameters = lines[0].strip().split(', ')
					self.n_lags, self.d, self.q = int(readed_parameters[0]), int(readed_parameters[1]), int(readed_parameters[2])

				wall = int(len(values)*0.6)
				wall_val= int(len(values)*0.2)
				train, val, test, last_values = values[:wall, 0], values[wall:wall+wall_val,0], values[wall+wall_val:-1,0], values[-1,0]
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_arima(train, val, [], [], [], test, n_series, self.d, self.q, n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 5):
				if(parameters_file_name):
					try:
						f = open(parameters_file_name, 'r')
					except FileNotFoundError:
						raise Exception('No existe el archivo: %s' % parameters_file_name)
					lines = f.readlines()
					if(len(lines) > 1):
						raise Exception('File with parameters can\'t have more than 1 line ')
					readed_parameters = lines[0].strip().split(', ')
					self.lr, self.n_epochs = float(readed_parameters[0]), int(readed_parameters[1])

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.split_data(data)
				start = timer()
				rmse, _, y, y_hat, last = modelos.model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.lr, n_features, -1, scaler, last_values, 
																				calc_val_error, calc_test_error, verbosity)
				print('time elapsed: ', timer() - start)
				
				print('rmse: ', rmse)

				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last
			
			else:
				raise Exception('hyperparameters combination is not in the valid options.')

