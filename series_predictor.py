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
	def __init__(self, datos, id_model, original, time_steps,  data_train, original_cols, tune, select, max_vars, verbosity, parameters_file_name=None, max_evals=100, saved_model=False, model_file_name=None, returns=False):
		"""
			Constructor de la clase, se encarga de cargar o entrenar el modelo según lo especificado en los hyperparámetros.

			Parámetros:
			- datos -- Arreglo de numpy, arreglo con todos los datos incluidos los de test (esto solo es necesario para la selección de variables)
			- id_model -- Entero, id del modelo que se va a utilizar
			- original -- Booleano, indica si entenar con las varaibles originales o con las eleccionadas. True para entrenar con las originales.
			- time_steps -- Entero, número de periodos en el futuro a predecir
			- data_train -- Arreglo de numpy, arreglo de numpy con los datos de entrenamiento con la serie a predecir en la primera posición
			- original_cols -- lista que contiene los nombres de las variables originales, para mantener los nombres cuando se seleccionan variables
			- tune -- *booleano* o entero que define si se hace tuning de parametros
			- select -- *booleano* o entero que defien si se hace selección de variables
			- max_vars -- entero que denota la cantidad maxima de variables a seleccionar a la hora de hacer selección de variables
			- verbosity -- entero que denota el nivel de verbosidad de la ejecución, entre más alto más graficos o información se mostrará (tiene un límite diferente para cada algoritmo)
			- parameters_file_name -- *string* que contien el nomrbe del archivo con los parametros a leer, si no se especifica se ejecutara con los parámetros por *default*
			- max_evals -- entero con la cantidad de ejecuciones a la hora de hacer tuning de parámetros, si no se especifica es 100
			- saved_model -- Booleano, indica si se desea entrenar el modelo o cargar uno guardado. Si True se carga un modelo guardado, si False se entrena un nuevo modelo.
			- model_file_name -- *string*, con el nombre del archivo que contiene el modelo que se quiere cargar
			- returns -- Booleano, indica si se está trabajando con retornos o no (serie diferenciada). True es que si se trabaja con retornos.

		"""
		self.id_model = id_model
		self.original = original
		self.time_steps = time_steps
		self.data_train = data_train
		self.original_cols = original_cols
		self.retrain_cont = 0 # contador de iteraciones para reentrenar
		self.retrain_rate = 10 # taza de reentreno para modelos que se necesiten reentrenar
		self.evaluating = False # parámetro para saber si el paso de training se hace al principio o luego cuando se está evaluando
		self.returns = False

		if(self.id_model == 0 and self.time_steps > 0):
			if(time_steps == 1):
				if(self.original):
					self.n_rnn, self.n_dense, self.activation, self.drop_p = 0, 0, 1, 0.0 # parámetros de arquitectura de la red
					self.batch_size, self.lr, self.n_epochs, self.n_hidden, self.n_lags = 52, 0.4799370248396754, 33, 159, 28
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 26, 181, 39, 3
					
				else:
					self.n_rnn, self.n_dense, self.activation, self.drop_p = 0, 0, 1, 0.0 # parámetros de arquitectura de la red
					# for IDCOT3TR_Index
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 100, 269, 9
					# for IBOXIG Index
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 200, 269, 25
					# for IBOXHY_Index
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 200, 269, 25
					# for GBIEMCOR Index
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 200, 269, 9
					# for JPEICORE Index
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 200, 269, 25
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 200, 250, 15
					# for SPTR Index
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 81, 200, 269, 25
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 100, 102, 5, 50 # bayes optim
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 44, 32, 207, 22 # bayes optim 2
					# returns
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 10, 300, 50, 10
					# self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 79, 90, 44, 28 # bayes optim
					self.n_rnn, self.n_dense, self.activation, self.drop_p = 2, 1, 1, 0.1529 # bayes optim 2
					self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 75, 158, 45, 43 # bayes optim 2
			else:
				if(self.original):
					self.n_rnn, self.n_dense, self.activation, self.drop_p = 0, 0, 1, 0.0 # parámetros de arquitectura de la red
					self.batch_size, self.lr, self.n_epochs, self.n_hidden, self.n_lags = 52, 0.4799370248396754, 33, 159, 28
				else:
					self.n_rnn, self.n_dense, self.activation, self.drop_p = 1, 1, 0, 0.2303
					self.batch_size, self.n_epochs, self.n_hidden, self.n_lags = 99, 300, 52, 28
		
		elif(self.id_model == 1 and self.time_steps == 1):
			if(self.original):
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = 4, 762, 18, 3 # for selected features
			else:
				# self.n_lags, self.n_estimators, self.max_features, self.min_samples = 6, 80, 12, 4
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = 13, 749, 5, 1
		
		elif(self.id_model == 2 and self.time_steps == 1):
			if(self.original):
				self.n_lags, self.n_estimators, self.lr, self.max_depth = 4, 808, 0.33209425848535884, 3
			else:
				self.n_lags, self.n_estimators, self.lr, self.max_depth = 5, 916, 0.6378995385153448, 4
		
		elif(self.id_model == 3 and self.time_steps == 1):
			self.n_lags = 4
		
		elif(self.id_model == 4 and self.time_steps == 1):
			# self.n_lags, self.d, self.q = 3, 0, 6 # best
			# self.n_lags, self.d, self.q = 12, 0 , 1
			# self.n_lags, self.d, self.q = 5, 2, 1
			# self.n_lags, self.d, self.q = 11, 1, 2
			# self.n_lags, self.d, self.q = 1, 0, 7
			# self.n_lags, self.d, self.q = 2, 2, 11 # too slow

			self.n_lags, self.d, self.q = 1, 1, 2
			# self.n_lags, self.d, self.q = 3, 0, 6

		else:
			raise Exception('hyperparameters combination is not in the valid options.')

		self.model = self.train(datos, original_cols, tune, select, max_vars, verbosity, parameters_file_name, max_evals, saved_model, model_file_name)


	def train(self, datos, original_cols, tune, select, max_vars, verbosity, parameters_file_name=None, max_evals=100, saved_model=False, model_file_name=None):
		"""
		Función que recibe como entrada los datos de entrenamiento junto con los parámetros y retorna el modelo entrenado.

		Parámetros:
		- datos -- Arreglo de numpy, arreglo con todos los datos incluidos los de test (esto solo es necesario para la selección de variables)
		- original_cols -- lista que contiene los nombres de las variables originales, para mantener los nombres cuando se seleccionan variables
		- tune -- *booleano* o entero que define si se hace tuning de parametros
		- select -- *booleano* o entero que defien si se hace selección de variables
		- max_vars -- entero que denota la cantidad maxima de variables a seleccionar a la hora de hacer selección de variables
		- verbosity -- entero que denota el nivel de verbosidad de la ejecución, entre más alto más graficos o información se mostrará (tiene un límite diferente para cada algoritmo)
		- parameters_file_name -- *string* que contien el nomrbe del archivo con los parametros a leer, si no se especifica se ejecutara con los parámetros por *default*
		- max_evals -- entero con la cantidad de ejecuciones a la hora de hacer tuning de parámetros, si no se especifica es 100
		- saved_model -- Booleano, indica si se desea entrenar el modelo o cargar uno guardado. Si True se carga un modelo guardado, si False se entrena un nuevo modelo.
		- model_file_name -- *string*, con el nombre del archivo que contiene el modelo que se quiere cargar
		- returns -- Booleano, indica si se está trabajando con retornos o no (serie diferenciada). True es que si se trabaja con retornos.

		Retorna:
		- model -- Modelo(varios tipos), modelo entrenado

		"""

		if(select):
			self.original = 0
			# feature selection
			import feature_selection
			# feature_selection.select_features_sa(pd.DataFrame(datos), max_vars)
			feature_selection.select_features_ga(pd.DataFrame(datos), max_vars, original_cols)
		
		if(not self.original and not self.evaluating):
			df = pd.read_csv('data/data_selected.csv', header=0, index_col=0)
			# df = pd.read_csv('data/data_selected_SPTR.csv', header=0, index_col=0)
			# df = pd.read_csv('data/forecast-competition-complete_selected.csv', index_col=0)
			# df = pd.read_csv('data/forecast-competition-complete_selected_manually.csv', index_col=0)
			self.data_train = df.values
			self.selected_features = list(df.columns)

		self.MAX_EVALS = max_evals
		values, self.scaler = utils.normalize_data(self.data_train, scale=(-1, 1))
		n_features = values.shape[1]

		calc_val_error = False if verbosity < 2 else True
		calc_test_error = True
		# calc_val_error = True
		# calc_test_error = False
		if(tune):
			best = utils.bayes_optimization(self.id_model, self.MAX_EVALS, values, self.scaler, n_features, self.time_steps, self.original, verbosity, model_file_name, self.returns)

			if(self.id_model == 0):
				if(model_file_name == None): model_file_name = 'models/lstm_%dtimesteps.h5' % (self.time_steps)
				# Parameters
				self.n_lags, self.n_epochs, self.batch_size, self.n_hidden = int(best['n_lags']), int(best['n_epochs']), int(best['batch_size']), int(best['n_hidden'])
				self.n_rnn, self.n_dense, self.activation, self.drop_p = int(best['n_rnn']), int(best['n_dense']), int(best['activation']), best['drop_p']
				f = open('parameters/optimized_lstm_%dtimesteps.pars' % self.time_steps, 'w')
				f.write('%d, %d, %d, %d\n' % (self.n_lags, self.n_epochs, self.batch_size, self.n_hidden))
				f.close()
				# train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 1)
				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values_without_val(values, self.n_lags, self.time_steps, 1)
				
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.n_epochs, self.batch_size, self.n_hidden, n_features, self.n_lags, 
																self.scaler, last_values, calc_val_error, calc_test_error, verbosity, saved_model, model_file_name, self.n_rnn, self.n_dense, self.activation, self.drop_p)
				
				print('rmse: %s ' % rmse)

				print('direction accuracy: %f%%' % (dir_acc*100))
				
				if(verbosity > 1):
					if(self.time_steps > 1):
						utils.plot_data_lagged_blocks([y_valset[:, 0].ravel(), y_hat_val], ['y', 'y_hat'], 'Validation plot')
					else:
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					if(self.time_steps > 1):
						utils.plot_data_lagged_blocks([y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
					else:
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 1 and self.time_steps == 1):
				if(model_file_name == None): model_file_name = 'models/randomForest.joblib'
				self.n_lags, self.n_estimators, self.max_features, self.min_samples = int(best['n_lags']), int(best['n_estimators']), int(best['max_features']), int(best['min_samples'])
				f = open('parameters/optimized_randomForest_%dtimesteps.pars' % self.time_steps, 'w')
				f.write('%d, %d, %d, %d\n' % (self.n_lags, self.n_estimators, self.max_features, self.min_samples))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.n_estimators, self.max_features, self.min_samples, 
																		n_features, self.n_lags, self.scaler, last_values, calc_val_error, calc_test_error, verbosity, saved_model, model_file_name)
				
				print('rmse: %s ' % rmse)

				print('direction accuracy: %f%%' % (dir_acc*100))

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 2 and self.time_steps == 1):
				if(model_file_name == None): model_file_name = 'models/adaBoost.joblib'
				self.n_lags, self.n_estimators, self.lr, self.max_depth = int(best['n_lags']), int(best['n_estimators']), best['lr'], best['max_depth']
				f = open('parameters/optimized_adaBoost_%dtimesteps.pars' % self.time_steps, 'w')
				f.write('%d, %d, %f\n' % (self.n_lags, self.n_estimators, self.lr))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.n_estimators, self.lr, self.max_depth, n_features, 
																	self.n_lags, self.scaler, last_values, calc_val_error, calc_test_error, verbosity, saved_model, model_file_name)

				print('rmse: %s ' % rmse)

				print('direction accuracy: %f%%' % (dir_acc*100))

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 3 and self.time_steps == 1):
				if(model_file_name == None): model_file_name = 'models/svm.joblib'
				self.n_lags = int(best['n_lags'])
				f = open('parameters/optimized_SVM_%dtimesteps.pars' % self.time_steps, 'w')
				f.write('%d\n' % (self.n_lags))
				f.close()

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, n_features, self.n_lags, self.scaler, last_values, calc_val_error, 
															calc_test_error, verbosity, saved_model, model_file_name)
				
				print('rmse: %s ' % rmse)

				print('direction accuracy: %f%%' % (dir_acc*100))

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 4 and self.time_steps == 1):
				if(model_file_name == None): model_file_name = 'models/arima.pkl'
				self.n_lags, self.d, self.q = int(best['n_lags']), int(best['d']), int(best['q'])
				f = open('parameters/optimized_arima_%dtimesteps.pars' % self.time_steps, 'w')
				f.write('%d, %d, %d\n' % (self.n_lags, self.d, self.q))
				f.close()

				wall = int(len(values)*0.6)
				wall_val= int(len(values)*0.2)
				train_X, val_X, test_X, last_values = values[:wall, :], values[wall:wall+wall_val,:], values[wall+wall_val:-1,:], values[-1,:]
				train_y, val_y, test_y = values[1:wall+1,0], values[wall+1:wall+wall_val+1,0], values[wall+wall_val+1:,0]
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_arima(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.d, self.q, n_features, self.n_lags, self.scaler, last_values, calc_val_error, 
																calc_test_error, verbosity, saved_model, model_file_name)
				print('time elapsed: ', timer() - start)

				print('rmse: %s ' % rmse)

				print('direction accuracy: %f%%' % (dir_acc*100))

				if(verbosity > 1):
					utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

				if(verbosity > 0):
					utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			else:
				raise Exception('hyperparameters combination is not in the valid options.')
		
		else:
			if(self.id_model == 0):
				if(model_file_name == None): model_file_name = 'models/lstm_%dtimesteps.h5' % (self.time_steps)

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

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 1)
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.n_epochs, self.batch_size, 
																									self.n_hidden, n_features, self.n_lags, self.scaler, last_values, calc_val_error, calc_test_error, 
																									verbosity, saved_model, model_file_name, self.n_rnn, self.n_dense, self.activation, self.drop_p, self.returns)
				print('time elapsed: ', timer() - start)

				if(not saved_model):
					print('rmse: %s ' % rmse)

					print('direction accuracy: %f%%' % (dir_acc*100))

					if(verbosity > 1):
						if(self.time_steps > 1):
							utils.plot_data_lagged_blocks([y_valset[:, 0].ravel(), y_hat_val], ['y', 'y_hat'], 'Validation plot')
						else:
							utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

					if(verbosity > 0):
						if(self.time_steps > 1):
							utils.plot_data_lagged_blocks([test_y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
						else:
							utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 1 and self.time_steps == 1):
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

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.n_estimators, self.max_features, self.min_samples, 
																		n_features, self.n_lags, self.scaler, last_values, calc_val_error, calc_test_error, verbosity, saved_model, model_file_name)
				print('time elapsed: ', timer() - start)

				if(not saved_model):
					print('rmse: %s ' % rmse)

					print('direction accuracy: %f%%' % (dir_acc*100))

					if(verbosity > 1):
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

					if(verbosity > 0):
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 2 and self.time_steps == 1):
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

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.n_estimators, self.lr, self.max_depth, n_features, 
																	self.n_lags, self.scaler, last_values, calc_val_error, calc_test_error, verbosity, saved_model, model_file_name)
				
				print('time elapsed: ', timer() - start)

				if(not saved_model):
					print('rmse: %s ' % rmse)

					print('direction accuracy: %f%%' % (dir_acc*100))

					if(verbosity > 1):
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

					if(verbosity > 0):
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 3 and self.time_steps == 1):
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

				train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
				start = timer()
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, n_features, self.n_lags, self.scaler, last_values, calc_val_error, 
																	calc_test_error, verbosity, saved_model, model_file_name)
				print('time elapsed: ', timer() - start)

				if(not saved_model):
					print('rmse: %s ' % rmse)

					print('direction accuracy: %f%%' % (dir_acc*100))

					if(verbosity > 1):
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

					if(verbosity > 0):
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			elif(self.id_model == 4 and self.time_steps == 1):
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
				rmse, _, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_arima(train_X, val_X, test_X, train_y, val_y, test_y, self.time_steps, self.d, self.q, n_features, self.n_lags, self.scaler, last_values, calc_val_error, 
																calc_test_error, verbosity, saved_model, model_file_name)

				if(model==None):
					raise Exception('Parámetros para SARIMAX invalidos, por lo que no se puede calcular')
				print('time elapsed: ', timer() - start)

				if(not saved_model):
					print('rmse: %s ' % rmse)

					print('direction accuracy: %f%%' % (dir_acc*100))

					if(verbosity > 1):
						utils.plot_data([y_valset, y_hat_val], ['y', 'y_hat'], 'Validation plot')

					if(verbosity > 0):
						utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')

				return model
			
			else:
				raise Exception('hyperparameters combination is not in the valid options.')

	def predict(self, new_ob):
		"""
			Función que predice con el modelo previamente entrenado dado un o unos nuevos valores. Con estos nuevos valores hace una iteración de entrenamiento antes de predecir

			Parámetros:
			- new_ob -- Arreglo de numpy, arreglo con la o las nuevas observaciones.

			Retorna:
			- last -- Arreglo de numpy | Lista | Entero, predicciones de los últimos valores dados, predicciones *out of sample*

		"""
		n_news = len(new_ob)
		self.evaluating = True
		if(self.data_train.shape[1] != new_ob.shape[1]):
			# if selected features
			new_ob = pd.DataFrame(new_ob)
			new_ob.columns = self.original_cols
			new_ob = new_ob[self.selected_features].values
		
		self.data_train = np.append(self.data_train, new_ob, axis=0)
		values = self.scaler.transform(self.data_train)
		
		if(self.id_model==0):
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 1)
			self.model.fit(test_X[[-n_news]], test_y[[-n_news]], epochs=10, batch_size=1, verbose=0, shuffle=False)
			last_values = np.expand_dims(last_values, axis=0)
			pred = self.model.predict(last_values).reshape(n_news, -1)

		elif(self.id_model==1):
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
			if(self.retrain_cont == self.retrain_rate):
				self.model = self.train(None, None, False, False, 0, 0, None, 0, False, None)
				self.retrain_cont = 0
			self.retrain_cont += 1
			self.model.set_params(n_jobs=1)
			pred = self.model.predict(last_values).reshape(n_news, -1)
		elif(self.id_model==2):
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
			if(self.retrain_cont == self.retrain_rate):
				self.model = self.train(None, None, False, False, 0, 0, None, 0, False, None)
				self.retrain_cont = 0
			self.retrain_cont += 1
			pred = self.model.predict(last_values).reshape(n_news, -1)
		elif(self.id_model==3):
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, self.n_lags, self.time_steps, 0)
			if(self.retrain_cont == self.retrain_rate):
				self.model = self.train(None, None, False, False, 0, 0, None, 0, False, None)
				self.retrain_cont = 0
			self.retrain_cont += 1
			pred = self.model.predict(last_values).reshape(n_news, -1)
		elif(self.id_model==4):
			last_values =  values[-n_news:,:]
			print(last_values.shape)
			pred = self.model.predict(len(values)-n_news, len(values) - 1, exog=last_values[:, 1:], endog=last_values[:, 0])

		tmp = np.zeros((pred.shape[1], self.data_train.shape[1]))
		tmp[:, 0] = pred
		last = self.scaler.inverse_transform(tmp)[:, 0]

		return last


