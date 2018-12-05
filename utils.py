import pandas as pd
import numpy as np 
import csv
import modelos

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

from timeit import default_timer as timer

ID_TO_MODELNAME = {0:'lstm', 1:'randomForest', 2:'adaBoost', 3:'svm', 4:'arima', 5:'lstmNoSW'}

def inverse_transform(data, scaler, n_features):
	data = data.copy()
	assert type(data) == np.ndarray
	if(data.ndim == 1): data = data.reshape(-1, 1)
	assert data.ndim == 2
	for i in range(data.shape[1]):
		tmp = np.zeros((data.shape[0], n_features))
		tmp[:, 0] = data[:, i]
		data[:, i] = scaler.inverse_transform(tmp)[:, 0]
	return data

def normalize_data(values, scale=(0,1)):
	"""
		Función para normalizar los datos, es decir, escalarlos en una escala que por *default* es [-1, 1]

		Parámetros:
		- values -- Arreglo de numpy, los datos
		- scaler -- Tupla de 2 valores, escala a la cual se quiere escalar los datos

		Retorna:
		- scaled -- Arreglo de numpy, los datos escalados
		- scaler -- Instancia de MinMaxScaler, para luego revertir el proceso de escalamiento
	"""
	# Be sure all values are numbers
	values = values.astype('float32')
	# scale the data
	scaler = MinMaxScaler(feature_range=scale)
	scaled = scaler.fit_transform(values)
	return scaled, scaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
		Función que convierte la serie de tiempo en datos supervisados para el modelo, es decir cambia el formato de (número de ejemplos, número de *features*) 
		por (número de ejemplos, número de *lags*, número de *features*)

		Parámetros:
		- data -- Arreglo de numpy, la serie completa de los datos
		- n_in -- Entero, número de *lags* o resagos de tiempo, *default* es 1
		- n_out -- Entero, número de *time steps* a predecir en el futuro, *default* es 1
		- dropnan -- Booleano, indica si se eliminan los valores de Nan del *dataframe* resultante

		Retorna:
		- agg -- *Dataframe* de pandas, *dataframe* con todas las variables en el nuevo formato, sus nombres de columnas son del tipo: "var3(t+2)"

	"""
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

def transform_values(data, n_lags, n_series, dim):
	"""
		Función para preprocesar la serie de entrada, primero le cambia el formato a (número de ejemplos, número de *lags*, número de *features*), luego divide estos datos en 
		entrenamiento, validación y *testing* adicionalmente retorna los ultimos valores para la predicción.

		Parámetros:
		- data -- Arreglo de numpy, serie de observaciones ordenada según orden cronológico
		- n_lags -- Entero, el número de *lags* que se usaran para entrenar
		- n_series -- Entero, el número de *time steps* a predecir en el futuro
		- dim -- Booleano, denota si se reformatea los valores de enrtenamiento resultantes, es decir si se cambia el formato de (número de ejemplos, número de *lags* X número de *features*) a (número de ejemplos, número de *lags*, número de *features*)

		Retorna:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de valdiación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de tiempos futuros de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de tiempos futuros de validación
		- test_y -- Arreglo de numpy, observaciones de tiempos futuros de *testing
		- last_values -- Arreglo de numpy, últimos datos apra hacer la predicción *out of sample*


	"""
	train_size = 0.6
	val_size =0.2
	test_size = 0.2
	n_features = data.shape[1]
	#print(data)
	#raise Exception('Debug')
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
	cols = ['var1(t)']
	cols += ['var1(t+%d)' % (i) for i in range(1, n_series)]
	y_o = reframed[cols].values
	train_o = y_o[:n_train]
	val_o = y_o[n_train:n_train + n_val]
	test_o = y_o[n_train + n_val:]

	train_X, train_y = train[:, :n_obs], train_o[:, -n_series:]
	val_X, val_y = val[:, :n_obs], val_o[:, -n_series:]
	test_X, test_y = test[:, :n_obs], test_o[:, -n_series:]

	# reshape train data to be 3D [n_examples, n_lags, features]
	if(dim):
		train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
		val_X = val_X.reshape((val_X.shape[0], n_lags, n_features))
		test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
		last_values = np.append(test_X[-1,1:n_lags,:], [test[-1,-n_features:]], axis=0)
	else:
		last_values = np.append(test_X[-1, n_features:].reshape(1,-1), [test[-1,-n_features:]], axis=1)	
		# return train_X, test_X, train_y, test_y, last_values
	return train_X, val_X, test_X, train_y, val_y, test_y, last_values


def plot_data(data, labels, title):
	"""
		Función para graficar los datos resultantes, solo sirve para las predicciones a 1 *time step*

		Parámetros:
		- data -- Lista de dos valores, lista con las predicciones y las observaciones
		- labels -- Lista de dos valores, lista con las etiquetas de los datos para mostrar en el gráfico
		- title -- String, título del gráfico

		Retorna:
		NADA

	"""
	plt.figure()
	for i in range(len(data)):
		plt.plot(data[i], label=labels[i])
	plt.suptitle(title, fontsize=16)
	plt.legend()
	plt.show()

def plot_data_lagged_blocks(data, labels, title):
	"""
		Función para graficar los datos resultantes, sirve más que todo para las predicciones a más de un *time step*. Lo que hace es ponerle un *padding* a las predicciones
		para que queden en el tiempo que están prediciendo, es decir, si *time steps* es 10, la priemra predicción se hara para el tiempo 0 y tendrá las predicciones hasta el
		tiempo 9, la segunda apredicción se hará en el tiempo 10 y tendra las predicciones del tiempo 10 al 19, etc.

		Parámetros:
		- data -- Lista de dos valores, lista con las predicciones y las observaciones, es necesario que la priemra posición sean las observaciones y la segunda las predicciones
		- labels -- Lista de dos valores, lista con las etiquetas de los datos para mostrar en el gráfico
		- title -- String, título del gráfico

		Retorna:
		NADA
	
	"""
	plt.figure()
	plt.plot(data[0], label=labels[0])
	for i in range(0, len(data[1]), len(data[1][0])):
		padding = [None for _ in range(i)]
		plt.plot(padding + list(data[1][i]), label=labels[1] + str(i+1))
	plt.suptitle(title, fontsize=16)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
	plt.show()

def diff(values):
	"""
		Función que sirve para diferenciar una serie
	
		Parámetros:
		- values -- Arreglo de numpy | lista, serie que va a ser diferenciada 

		Retorna:
		- new -- Lista, lista con la serie diferenciada

	"""
	new = np.zeros(len(values)-1)
	for i in range(len(new)):
		new[i] = values[i+1] - values[i]
	return new


def calculate_diff_level_for_stationarity(values, scaler, maxi):
	"""	
		Función que sirve para calcular el nivel de diferenciación necesario para que una serie sea estacionaria
		
		Parámetros:
		- values -- Arreglo de numpy, serie sobre la cual se va a calcular el nivel de diferenciación
		- scaler -- Instancia de la clase MinMaxScaler de sklearn, sirve para revertir el escalamiento de la serie
		- maxi -- Entero, valor maximo de diferenciación permitido, si se llega a este limite y la serie no es estacionaria se devolverá maxi como el nivel de diferenciación

		Retorna:
		- maxi | i -- Entero, minimo nivel de diferenciación para que la serie sea estacionaria, o maximo número en el que se diferencio en el caso de que la serie no se logro poner estacionaria

	"""
	from statsmodels.tsa.stattools import adfuller

	real_values = scaler.inverse_transform(values)
	serie = real_values[:, 0]
	for i in range(maxi):
		result = adfuller(serie)
		if(result[0] < result[4]['5%']):
			return i
		serie = diff(serie)
	return maxi

def objective(params, id_model, values, scaler, n_features, n_series, verbosity, model_file_name):
	"""
		Función objetivo que sirve para la optimización bayesiana, sirve para ejecuar el modelo con los parámetros recibidos, calcular el error de esta ejecución y así decidir
		cuales parámetros son mejores.
		Parámetros:
		- params -- Diccionario, contien los parametros para la ejecución, estos parámetros son dados por la libreria de optimización bayesiana (hyperopt) dentro de un espacio previamente definido
		- id_model -- Entero, id del modelo que se va a entrenar
		- values -- Arreglo de numpy, datos con los cuales se va a entrenar el modelo, es decir, la serie previamente preprocesada
		- scaler -- Instancia de la clase MinMaxScaler de sklearn, sirve para el escalamiento de los datos y para revertir este escalamiento
		- n_features -- Entero, número de *features* de la serie
		- n_series -- Entero, número de *time steps*
		- verbosity -- Entero, nivel de verbosidad de la ejecución
		- model_file_name -- String, nombre del archivo donde se guardará y/o se cargará el modelo entrenado
		Retorna:
		- [valor] -- Flotante, dicconario qu retorna la "recompensa" de esta ejecuión, la idea esmaximizarla, por eso el "error" de la ejecución se invierte
	"""
	calc_val_error = True
	calc_test_error = True
	if(id_model == 0):
		if(model_file_name == None): model_file_name = 'models/trials-lstm.h5'

		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_epochs', 'batch_size', 'n_hidden', 'n_dense', 'n_rnn', 'activation']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 1)
		rmse, rmse_val, _, _, _, _, _, dir_acc, _ = modelos.model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params['n_epochs'], params['batch_size'], params['n_hidden'], n_features, 
														params['n_lags'], scaler, last_values, calc_val_error, calc_test_error, verbosity, False, model_file_name, params['n_rnn'], params['n_dense'], params['activation'], params['drop_p'])
		mean = np.mean(test_y)
		rmse = np.abs((mean - rmse_val)/mean) + (1 - dir_acc)
		run_time = timer() - start
		# print('no_score: ', rmse)
		# print('time: ', run_time, end='\n\n')
	elif(id_model == 1):
		if(model_file_name == None): model_file_name = 'models/trials-randomForest.joblib'

		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_estimators', 'max_features', 'min_samples']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 0)
		rmse, rmse_val, _, _, _, _, _, dir_acc, _ = modelos.model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params['n_estimators'], params['max_features'], params['min_samples'], 
																n_features, params['n_lags'], scaler, last_values, calc_val_error, calc_test_error, verbosity, False, model_file_name)
		mean = np.mean(test_y)
		rmse = np.abs((mean - rmse_val)/mean) + (1 - dir_acc)
		run_time = timer() - start
		# print('no_score: ', rmse)
		# print('time: ', run_time, end='\n\n')
	elif(id_model == 2):
		if(model_file_name == None): model_file_name = 'models/trials-adaBoost.joblib'
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_estimators', 'max_depth']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 0)
		rmse, rmse_val, _, _, _, _, _, dir_acc, _ = modelos.model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params['n_estimators'], params['lr'], params['max_depth'], n_features, 
															params['n_lags'], scaler, last_values, calc_val_error, calc_test_error, verbosity, False, model_file_name)
		mean = np.mean(test_y)
		rmse = np.abs((mean - rmse_val)/mean) + (1 - dir_acc)
		run_time = timer() - start
		# print('no_score: ', rmse)
		# print('time: ', run_time, end='\n\n')
	elif(id_model == 3):
		if(model_file_name == None): model_file_name = 'models/trials-svm.joblib'
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, val_X, test_X, train_y, val_y, test_y, last_values = transform_values(values, params['n_lags'], n_series, 0)
		rmse, rmse_val, _, _, _, _, _, dir_acc, _ = modelos.model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, params['n_lags'], scaler, last_values, calc_val_error, calc_test_error, 
													verbosity, False, model_file_name)
		mean = np.mean(test_y)
		rmse = np.abs((mean - rmse_val)/mean) + (1 - dir_acc)
		run_time = timer() - start
		# print('no_score: ', rmse)
		# print('time: ', run_time, end='\n\n')
	elif(id_model == 4):
		if(model_file_name == None): model_file_name = 'models/arima.pkl'
		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'd', 'q']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		wall = int(len(values)*0.6)
		wall_val= int(len(values)*0.2)
		train_X, val_X, test_X, last_values = values[:wall, :], values[wall:wall+wall_val,:], values[wall+wall_val:-1,:], values[-1,:]
		train_y, val_y, test_y = values[1:wall+1,0], values[wall+1:wall+wall_val+1,0], values[wall+wall_val+1:,0]
		start = timer()
		rmse, rmse_val, y, y_hat, y_valset, y_hat_val, last, dir_acc, model = modelos.model_arima(train_X, val_X, test_X, train_y, val_y, test_y, n_series, params['d'], params['q'], n_features, params['n_lags'], scaler, last_values, calc_val_error, 
														calc_test_error, verbosity, False, model_file_name)
		mean = np.mean(test_y)
		rmse = np.abs((mean - rmse_val)/mean) + (1 - dir_acc)
		run_time = timer() - start
		# print('no_score: ', rmse)
		# print('time: ', run_time, end='\n\n')

	return -1 * rmse

def bayes_optimization(id_model, MAX_EVALS, values, scaler, n_features, n_series, original, verbosity, model_file_name):
	"""
		Función para encontrar los parámetros optimos para un modelo
		Parámetros:
		- id_model -- Entero, id del modelo que se va a entrenar
		- MAX_EVALS -- Entero, número máximo  de iteraciones de la optimización bayesiana
		- values -- Arreglo de numpy, datos con los cuales se va a entrenar el modelo, es decir, la serie previamente preprocesada
		- scaler -- Instancia de la clase MinMaxScaler de sklearn, sirve para el escalamiento de los datos y para revertir este escalamiento
		- n_features -- Entero, número de *features* de la serie
		- n_series -- Entero, número de *time steps*
		- original -- Booleano, denota si se van a usar los *features* originales de la serie o los *features* seleccionados, en esta función serviría para identificar el archivo de salida
		- verbosity -- Entero, nivel de verbosidad de la ejecución
		- model_file_name -- String, nombre del archivo donde se guardará y/o se cargará el modelo entrenado
		Retorna: 
		- best -- Diccionario, diccionario con los mejores parámetros encontrados en la optimización bayesiana
	"""
	from bayes_opt import BayesianOptimization
	if(id_model == 0):
		space = {'activation': (0.1, 1.9),
				'batch_size': (5, 100),
				'drop_p': (0, 1),
				'n_dense': (0, 5),
				'n_epochs': (10, 200),
				'n_hidden': (5, 300),
				'n_lags': (1, (min(50, int(len(values)/2)))),
				'n_rnn': (0, 4)}
		func = lambda activation, batch_size, drop_p, n_dense, n_epochs, n_hidden, n_lags, n_rnn: objective({'activation':activation , 'batch_size':batch_size, 'drop_p':drop_p, 'n_dense':n_dense, 'n_epochs':n_epochs, 'n_hidden':n_hidden, 'n_lags':n_lags, 'n_rnn':n_rnn}, id_model, values, scaler, n_features, n_series, verbosity, model_file_name)
	elif(id_model == 1):
		space = {'max_features': (1, n_features),
				'min_samples': (1, 20),
				'n_estimators': (10, 1000),
				'n_lags': (1, min(50, int(len(values)/2)))}
		func = lambda max_features, min_samples, n_estimators, n_lags: objective({'max_features': max_features, 'min_samples': min_samples, 'n_estimators': n_estimators, 'n_lags': n_lags}, id_model, values, scaler, n_features, n_series, verbosity, model_file_name)
	elif(id_model == 2):
		space = {'lr': (0.00001, 1.0),
				'max_depth': (2, 10),
				'n_estimators': (10, 1000),
				'n_lags': (1, min(50, int(len(values)/2)))}
		func = lambda lr, max_depth, n_estimators, n_lags: objective({'lr':lr, 'max_depth': max_depth, 'n_estimators': n_estimators, 'n_lags': n_lags}, id_model, values, scaler, n_features, n_series, verbosity, model_file_name)
	elif(id_model == 3):
		space = {'n_lags': (1, min(50, int(len(values)/2)))}
		func = lambda  n_lags: objective({'n_lags': n_lags}, id_model, values, scaler, n_features, n_series, verbosity, model_file_name)
	elif(id_model == 4):
		diff_level = calculate_diff_level_for_stationarity(values, scaler, 5)
		space={'n_lags': (1, 12),
				'q': (1, 12)}
		func = lambda  n_lags, q: objective({'d': diff_level, 'n_lags': n_lags, 'q': q}, id_model, values, scaler, n_features, n_series, verbosity, model_file_name)

	optimizer = BayesianOptimization(f=func, pbounds=space, verbose=2, random_state=np.random.randint(np.random.randint(100)))
	if(id_model == 0):
		optimizer.probe(params={'activation':1.0, 'batch_size':10.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':300.0, 'n_hidden':50.0, 'n_lags':10.0, 'n_rnn':0.0})
		optimizer.probe(params={'activation':1.0, 'batch_size':81.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':200.0, 'n_hidden':250.0, 'n_lags':15.0, 'n_rnn':0.0})
		optimizer.probe(params={'activation':1.0, 'batch_size':81.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':200.0, 'n_hidden':269.0, 'n_lags':9.0, 'n_rnn':0.0})
		optimizer.probe(params={'activation':1.0, 'batch_size':81.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':200.0, 'n_hidden':269.0, 'n_lags':25.0, 'n_rnn':0.0})
	optimizer.maximize(init_points=2, n_iter=MAX_EVALS)


	# store best results
	best = optimizer.max['params']
	of_connection = open('trials/bests.txt', 'a')
	writer = csv.writer(of_connection)
	if(id_model == 0):
		writer.writerow([optimizer.max['target'], best['activation'], best['batch_size'], best['drop_p'], best['n_dense'], best['n_epochs'], best['n_hidden'], obest['n_lags'], best['n_rnn'], MAX_EVALS])
	elif(id_model == 1):
		writer.writerow([optimizer.max['target'], best['n_lags'], best['n_estimators'], best['max_features'], best['min_samples'], MAX_EVALS])
	elif(id_model == 2):
		writer.writerow([optimizer.max['target'], best['n_lags'], best['n_estimators'], best['lr'], best['max_depth'], MAX_EVALS])
	elif(id_model == 3):
		writer.writerow([optimizer.max['target'], best['n_lags'], MAX_EVALS])
	elif(id_model == 4):
		best.update({'d': diff_level})
		writer.writerow([optimizer.max['target'], best['d'], best['n_lags'], best['q'], MAX_EVALS])
	of_connection.close()

	return best




def get_direction_accuracy(y, y_hat):
	"""
		Función para calcular el % de aciertos en la dirección, teniendo en cuenta las observaciones y las predicciones.
		Por dirección se entiende que si en la observación el valor sube, en la predicción también igualmente si baja.

		Parámetros:
		- y -- Arreglo de numpy, las observaciones
		- y_hat -- Arreglo de numpy, las predicciones

		Retorna:
		- [valor] -- % de aciertos en la dirección
	"""
	assert len(y) == len(y_hat)
	y_dirs = [1 if y[i] < y[i+1] else 0 for i in range(len(y) - 1)]
	y_hat_dirs = [1 if y_hat[i] < y_hat[i+1] else 0 for i in range(len(y_hat) - 1)]
	return sum(np.array(y_dirs) == np.array(y_hat_dirs))/len(y)

def get_returns_direction_accuracy(y, y_hat):
	"""
		Función para calcular el % de aciertos en la dirección de retornos, teniendo en cuenta las observaciones y las predicciones.
		Por dirección se entiende que si en la observación el valor sube, en la predicción también lo haga. De igual forma en el caso de que baje.

		Parámetros:
		- y -- Arreglo de numpy, las observaciones
		- y_hat -- Arreglo de numpy, las predicciones

		Retorna:
		- [valor] -- % de aciertos en la dirección
	"""
	assert len(y) == len(y_hat)
	y_dirs = [1 if y[i]>0 else 0 for i in range(len(y))]
	y_hat_dirs = [1 if y_hat[i]>0 else 0 for i in range(len(y_hat))]
	return sum(np.array(y_dirs) == np.array(y_hat_dirs))/len(y)

def get_returns_values_direction_accuracy(values, returns):
	"""
		Función para calcular el % de aciertos en la dirección de retornos teniendo una serie en valores y otra en retornos, teniendo en cuenta las observaciones y las predicciones.
		Por dirección se entiende que si en la observación el valor sube, en la predicción también lo haga. De igual forma en el caso de que baje.

		Parámetros:
		- values -- Arreglo de numpy, serie de valores
		- returns -- Arreglo de numpy, serie de retornos
		Retorna:
		- [valor] -- % de aciertos en la dirección
	"""
	assert len(values) == len(returns) + 1
	values_dirs = [1 if values[i] < values[i+1] else 0 for i in range(len(values) - 1)]
	returns_dirs = [1 if returns[i]>0 else 0 for i in range(len(returns))]
	return sum(np.array(values_dirs) == np.array(returns_dirs))/len(returns)

def calculate_rmse(y, y_hat):
	"""
		Función que calcula el rmse (raiz del error medio cuadrático) entre dos arreglos de datos

		Parámetros:
		- y -- Arreglo de numpy | Lista, serie de observaciones o valores reales
		- y_hat -- Arreglo de numpy | Lista, serie de predicciones 
	"""
	assert len(y) == len(y_hat)
	from sklearn.metrics import mean_squared_error
	return np.sqrt(mean_squared_error(y, y_hat))


