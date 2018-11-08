import numpy as np
import pandas as pd
import math
import utils

import modelos

from timeit import default_timer as timer


class Predictor():
	"""
		Clase para crear los modelos, entrenarlos y predecir con ellos, en general es la clase principal y mediante esta se interactua con los modelos.

	"""
	def __init__(self, id_model, original, time_steps):
		self.id_model = id_model
		self.original = original
		self.time_steps = time_steps
		if(self.id_model == 0 and time_steps > 0):
			if(self.original):
				self.batch_size, self.lr, self.n_epochs, self.n_hidden, self.n_lags = 52, 0.4799370248396754, 33, 159, 28 
				# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 26, 181, 39, 3
			else:
				# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 30, 7, 288, 65 # 15, 91, 24, 2
				# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 51, 21, 108, 35
				self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 30, 17, 88, 41
				#self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 30, 47, 188, 41
				# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 10, 47, 188, 41
		
		elif(self.id_model == 1 and time_steps == 1):
			if(self.original):
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = 4, 762, 18, 3 # for selected features
			else:
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = 6, 80, 12, 4
		
		elif(self.id_model == 2 and time_steps == 1):
			if(self.original):
				self.n_lags, self.n_estimators, self.lr, self.max_depth = 4, 808, 0.33209425848535884, 3
			else:
				self.n_lags, self.n_estimators, self.lr, self.max_depth = 5, 916, 0.6378995385153448, 4
		
		elif(self.id_model == 3 and time_steps == 1):
			self.n_lags = 4
		
		elif(self.id_model == 4 and time_steps == 1):
			self.n_lags, self.d, self.q = 3, 0, 6 # best
			# self.n_lags, self.d, self.q = 12, 0 , 1
			# self.n_lags, self.d, self.q = 5, 2, 1
			# self.n_lags, self.d, self.q = 11, 1, 2
			# self.n_lags, self.d, self.q = 1, 0, 7
			# self.n_lags, self.d, self.q = 2, 2, 11 # too slow

			# self.n_lags, self.d, self.q = 0, 0, 2

		
		elif(self.id_model == 5 and time_steps > 0):
			if(self.original):
				self.lr, self.n_epochs = 7.389422743950269e-05, 182 # 0.014540077212290003, 37
			else:
				self.lr, self.n_epochs = 0.2050774898644015, 22
		else:
			raise Exception('hyperparameters combination is not in the valid options.')


	def predict(self, data, original_cols, tune, select, max_vars, verbosity, parameters_file_name=None, max_evals=100, only_predict=False, model_file_name=None):
		"""
		Función que recibe como entrada los datos de entrenamiento junto con los parámetros y retorna las predicciones a los *timesteps* especificados.

		Parámetros:
		- data -- arreglo de numpy con los datos de entrenamiento con la serie a predecir en la primera posición
		- original_cols -- lista que contiene los nombres de las variables originales, para mantener los nombres cuando se seleccionan variables
		- tune -- *booleano* o entero que define si se hace tuning de parametros
		- select -- *booleano* o entero que defien si se hace selección de variables
		- max_vars -- entero que denota la cantidad maxima de variables a seleccionar a la hora de hacer selección de variables
		- verbosity -- entero que denota el nivel de verbosidad de la ejecución, entre más alto más graficos o información se mostrará (tiene un límite diferente para cada algoritmo)
		- parameters_file_name -- *string* que contien el nomrbe del archivo con los parametros a leer, si no se especifica se ejecutara con los parámetros por *default*
		- max_evals -- entero con la cantidad de ejecuciones a la hora de hacer tuning de parámetros, si no se especifica es 100
		- only_predict -- *booleano* que denota si solo se quiere predecir utilizando un modelo previamente entrenado, si no se especifica es *false*
		- model_file_name -- *string* con el nombre del archivo que contiene el model oque se quiere leer para solamente hacer las predicciones

		Retorna:
		- last -- Arreglo de numpy | Lista | Entero, predicciones de los últimos valores dados, predicciones *out of sample*

		"""

		if(select):
			self.original = 0
			# feature selection
			import feature_selection
			# feature_selection.select_features_sa(pd.DataFrame(data), max_vars)
			feature_selection.select_features_ga(pd.DataFrame(data), max_vars, original_cols)
			# df = pd.read_csv('data/forecast-competition-complete.csv', index_col=0, header=0)
			# feature_selection.select_features_stepwise_forward(df, max_vars)
		
		if(not self.original):
			df = pd.read_csv('data/forecast-competition-complete_selected.csv', index_col=0)
			# df = pd.read_csv('data/forecast-competition-complete_selected_manually.csv', index_col=0)
			data = df.values

		self.MAX_EVALS = max_evals
		values, scaler = utils.normalize_data(data)
		n_series = self.time_steps
		n_features = values.shape[1]

		calc_val_error = False if verbosity < 2 else True
		calc_test_error = True
		if(tune):
			best, results = utils.bayes_optimization(self.id_model, self.MAX_EVALS, values, scaler, n_features, n_series, self.original, verbosity, model_file_name)

			if(self.id_model == 0):
				if(model_file_name == None): model_file_name = 'models/lstm_%dtimesteps.h5' % (n_series)
				# Parameters
				self.n_lags, self.n_epochs, self.batch_size, self.n_hidden = int(best['n_lags']), int(best['n_epochs']), int(best['batch_size']), int(best['n_hidden'])
				f = open('parameters/optimized_lstm_%dtimesteps.pars' % n_series, 'w')
				f.write('%d, %d, %d, %d\n' % (self.n_lags, self.n_epochs, self.batch_size, self.n_hidden))
				f.close()
				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 1)
				
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.batch_size, self.n_hidden, n_features, self.n_lags, 
																scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)
				
				print('rmse: %s ' % rmse)
				
				if(verbosity > 1):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([y_valset[:, 0].ravel(), y_hat_val], ['y', 'y_hat'], 'Validation plot')
					else:
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 1 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/randomForest.joblib'
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = int(best['n_lags']), int(best['n_estimators']), int(best['max_features']), int(best['min_samples'])
				f = open('parameters/optimized_randomForest_%dtimesteps.pars' % n_series, 'w')
				f.write('%d, %d, %d, %d\n' % (self.n_lags, self.n_estimators, self.max_features, self.min_samples))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.max_features, self.min_samples, 
																		n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)
				
				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 2 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/adaBoost.joblib'
				self.n_lags, self.n_estimators, self.lr, self.max_depth = int(best['n_lags']), int(best['n_estimators']), best['lr'], best['max_depth']
				f = open('parameters/optimized_adaBoost_%dtimesteps.pars' % n_series, 'w')
				f.write('%d, %d, %f\n' % (self.n_lags, self.n_estimators, self.lr))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.lr, self.max_depth, n_features, 
																	self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)

				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 3 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/svm.joblib'
				self.n_lags = int(best['n_lags'])
				f = open('parameters/optimized_SVM_%dtimesteps.pars' % n_series, 'w')
				f.write('%d\n' % (self.n_lags))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, n_series, 0)
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, self.n_lags, scaler, last_values, calc_val_error, 
															calc_test_error, verbosity, only_predict, model_file_name)
				
				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 4 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/arima.pkl'
				self.n_lags, self.d, self.q = int(best['n_lags']), int(best['d']), int(best['q'])
				f = open('parameters/optimized_arima_%dtimesteps.pars' % n_series, 'w')
				f.write('%d, %d, %d\n' % (self.n_lags, self.d, self.q))
				f.close()

				wall = int(len(values)*0.6)
				wall_val= int(len(values)*0.2)
				train, val, test, last_values = values[:wall, 0], values[wall:wall+wall_val,0], values[wall+wall_val:-1,0], values[-1,0]
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_arima(train, val, [], [], [], test, n_series, self.d, self.q, n_features, self.n_lags, scaler, last_values, calc_val_error, 
																calc_test_error, verbosity, only_predict, model_file_name)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 5):
				if(model_file_name == None): model_file_name = 'models/lstm-noSW_%dtimesteps.h5' % (n_series)
				self.lr, self.n_epochs = best['lr'], int(best['n_epochs'])
				f = open('parameters/optimized_lstmNoSW_%dtimesteps.pars' % n_series, 'w')
				f.write('%f, %d\n' % (self.lr, self.n_epochs))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.split_data(values)
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.lr, n_features, scaler, 
																				last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)

				print('rmse: %s ' % rmse)
				
				if(verbosity > 1):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([y_valset[:, 0].ravel(), y_hat_val], ['y', 'y_hat'], 'Validation plot')
					else:
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			else:
				raise Exception('hyperparameters combination is not in the valid options.')
		
		else:
			if(self.id_model == 0):
				if(model_file_name == None): model_file_name = 'models/lstm_%dtimesteps.h5' % (n_series)

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
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.batch_size, self.n_hidden, n_features, 
																self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)
				print('time elapsed: ', timer() - start)
				
				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([y_valset[:, 0].ravel(), y_hat_val], ['y', 'y_hat'], 'Validation plot')
					else:
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 1 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/randomForest.joblib'
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
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.max_features, self.min_samples, 
																		n_features, self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 2 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/adaBoost.joblib'
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
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_estimators, self.lr, self.max_depth, n_features, 
																	self.n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)
				
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 3 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/svm.joblib'
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
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, self.n_lags, scaler, last_values, calc_val_error, 
																	calc_test_error, verbosity, only_predict, model_file_name)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 4 and n_series == 1):
				if(model_file_name == None): model_file_name = 'models/arima.pkl'
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
				train_X, val_X, test_X, last_values = values[:wall, :], values[wall:wall+wall_val,:], values[wall+wall_val:-1,:], values[-1,:]
				train_y, val_y, test_y = values[1:wall+1,0], values[wall+1:wall+wall_val+1,0], values[wall+wall_val+1:,0]
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_arima(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.d, self.q, n_features, self.n_lags, scaler, last_values, calc_val_error, 
																calc_test_error, verbosity, only_predict, model_file_name)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			elif(self.id_model == 5):
				if(model_file_name == None): model_file_name = 'models/lstm-noSW_%dtimesteps.h5' % (n_series)
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

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.split_data(values)
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last = modelos.model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, self.n_epochs, self.lr, n_features, scaler, 
																				last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name)
				print('time elapsed: ', timer() - start)
				
				print('rmse: ', rmse)

				if(verbosity > 1):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([y_valset[:, 0].ravel(), y_hat_val], ['y', 'y_hat'], 'Validation plot')
					else:
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					if(n_series > 1):
						utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return last.squeeze()
			
			else:
				raise Exception('hyperparameters combination is not in the valid options.')

