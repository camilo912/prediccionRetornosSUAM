from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import numpy as np
import math

def calculate_rmse(n_series, n_features, n_lags, X, y, scaler, model):
	import utils
	"""
		
		Función para calcular la raíz del error cuadrático medio, solo es utilizado por los algoritmos:
		*Random forest
		*Ada boost
		*SVM (máquinas de soporte vectorial o máquinas de soporte de vectores)
		 Los otros algoritmos tienen su propia forma de calcular este error.

		Parámetros:
		- n_series -- Entero, el número de time steps a predecir en el futuro, en estos metodos es siempre es 1 por ahora, hasta que se implementen para varios time steps
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaron para entrenar, estos *lags* también son conocidos como resagos. Intuitivamente se pueden enterder como una ventana deslizante o como cuantos tiempos atrás en el tiempo tomo en cuenta para hacer mi predicción
		- X -- Arreglo de numpy, los datos con los que se quier hacer la rpedicción para calcular el error
		- y -- Arreglo de numpy, las observaciones contra las cuales se van a comparar las predicciones para luego calcular el error
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente. En esta función se utiliza para revertir el escalamiento y obtener los datos reales
		- model -- Modelo de sklearn, el modelo entrenado con el cual se van a realizar las predicciones

		

		Retorna:
		- inv_y -- Arreglo de numpy, observaciones en la escala real
		- inv_yhat -- Arreglo de numpy, predicciones en la escala real
		- rmse -- Flotante, raíz del error curadrático medio entre las observaciones y las predicciones

	"""
	yhat = model.predict(X)
	# Xs = np.ones((X.shape[0], n_lags * n_features))
	# yhat = yhat.reshape(-1, n_series)
	# inv_yhats = []
	# for i in range(n_series):
	# 	inv_yhat = np.concatenate((Xs[:, -(n_features - 1):], yhat[:, i].reshape(-1, 1)), axis=1)
	# 	inv_yhat = scaler.inverse_transform(inv_yhat)

	# 	inv_yhat = inv_yhat[:, -1]
	# 	inv_yhats.append(inv_yhat)

	# inv_yhat = np.array(inv_yhats).T
	inv_yhat = utils.inverse_transform(yhat, scaler, n_features)
	
	# invert scaling for actual
	# y = y.reshape((len(y), n_series))
	# inv_ys = []
	# for i in range(n_series):
	# 	inv_y = np.concatenate((Xs[:, -(n_features - 1):], y[:, i].reshape(-1, 1)), axis=1)
	# 	inv_y = scaler.inverse_transform(inv_y)
	# 	inv_y = inv_y[:,-1]
	# 	inv_ys.append(inv_y)

	# inv_y = np.array(inv_ys).T
	inv_y = utils.inverse_transform(y, scaler, n_features)

	# calculate RMSE	
	rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
	return inv_y, inv_yhat, rmse

def predict_last(n_series, n_features, n_lags, X, scaler, model, dim):
	"""
		
		Función para predecir los últimos valores y sacar la predicción que se le retorna al usuario

		Parámetros:
		- n_series -- Entero, el número de time steps a predecir en el futuro, en estos metodos es siempre es 1 por ahora, hasta que se implementen para varios time steps
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaron para entrenar
		- X -- Arreglo de numpy, los datos con los que se quier hacer la rpedicción para calcular el error
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- model -- Modelo de sklearn, el modelo entrenado con el cual se van a realizar las predicciones
		- dim -- Booleano, Parámetro para controlar los problemas de formato, si se especifica este parametro en True se agrega una dimensión extra a los datos.

		Retorna:
		- inv_yhat -- Arreglo de numpy, predicción de los últimos valores en escala real

	"""
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
	"""
		Función para calcular el error personalizado del modelo LSTM, este error personalizado es la raíz del error cuadrático medio dandole más importancia a los primeros *lags*

		Parámetros:
		- yTrue -- valores con los cuales e va a comaprar las predicciones
		- yPred -- predicciones para calcular el error

		Retorna:
		- valor -- error cuadrático medio ponderado
	"""
	from keras import backend as K
	ones = K.ones_like(yTrue[0,:]) # a simple vector with ones shaped as (n_series,)
	idx = K.cumsum(ones) # similar to a 'range(1,n_series+1)'

	return K.mean((1/idx)*K.square(yTrue-yPred))

def model_lstm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_epochs, batch_size, n_hidden, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name):
	"""
		
		Función para crear, entrenar y calcular el error del modelo LSTM, también sirve para predecir
		Este modelo está hecho en Keras solamente tiene dos capas una de LSTM y otra Densa o totalmente conectada, utiliza el optimizador Adam y la función de error es personalizada para darle mayor importancia a los priemros *lags*

		Parámetros:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de validación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de validación
		- test_y -- Arreglo de numpy, observaciones de *testing*
		- n_series -- Entero, el número de time steps a predecir en el futuro
		- n_epochs -- Entero, número de epocas de entrenamiento
		- batch_size -- Entero, tamaño de los *bathcs* de entrenamiento
		- n_hidden -- Entero, número de estados escondidos de la red neuronal LSTM
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaran para entrenar
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- last_values -- Arreglo de numpy, últimos valores para realizar la predicción
		- calc_val_error -- Booleano, indica si se calcula el error de validación, si no se calcula se devuelve None en este campo
		- calc_test_error -- Booleano, indica si se calcula el error de *testing*, si no se calcula se devuelve None en este campo
		- verbosity -- Entero, nivel de verbosidad de la ejecución del modelo, entre más alto más información se mostrará el límite es 4 y debe ser mayor o igual a 0
		- only_predict -- Booleano, indica si solo se desea predecir o si también se desea entrenar el modelo. True sería solo predecir, False también entrenaría el modelo.
		- model_file_name -- *String*, nombre del archivo donde se guardará y/o se cargará el modelo.

		Retorna:
		- rmse -- Flotante, raíz del error medio cuadrático de *testing*; retorna None si calc_test_error es False
		- rmse_val -- Flotante, raíz del error medio cuadrático de validación; retorna None si calc_val_error es False
		- test_y -- Arreglo de numpy, observaciones de la particón de *testing* en escala real
		- y_hat_test -- Arreglo de numpy, predicciones de la partición de *testing* en escala real
		- val_y -- Arreglo de numpy, observaciones de la particón de validación en escala real
		- y_hat_val -- Arreglo de numpy, predicciones de la partición de validación en escala real
		- last -- arreglo de numpy, predicción de los últimos valores


	"""
	n_out = n_series
	
	if(not only_predict):
		print('training...')

		from keras.layers import Dense, Dropout, LSTM
		from keras.models import Sequential
		from keras.optimizers import Adam
		from keras import backend as K

		drop_p = 0.05
		verbose_dict = {0:0, 1:2, 2:1}
		verbose = 0 if verbosity < 3 else min(verbosity - 2, 2)
		verbose = verbose_dict[verbose]

		model = Sequential()
		if(n_series == 1):
			model.add(LSTM(n_hidden, input_shape=(n_lags, n_features), return_sequences=True))
			model.add(Dropout(0.25))
			model.add(LSTM(n_hidden))
			model.add(Dropout(0.75))
			model.add(Dense(n_out))
		else:
			model.add(LSTM(n_hidden, input_shape=(n_lags, n_features), return_sequences=True))
			model.add(Dropout(0.5))
			model.add(Dense(n_hidden, activation='relu'))
			model.add(Dropout(0.5))
			model.add(LSTM(n_hidden, return_sequences=True))
			model.add(Dropout(0.5))
			model.add(Dense(n_hidden, activation='relu'))
			model.add(Dropout(0.5))
			model.add(LSTM(n_hidden))
			model.add(Dropout(0.5))
			model.add(Dense(n_out))

		opt = Adam(lr=0.001)#, decay=0.0)
		model.compile(loss=weighted_mse, optimizer=opt)
		# model.compile(loss=lambda yTrue, yPred: K.mean((1/K.cumsum(K.ones_like(yTrue[0,:])))*K.square(yTrue-yPred)), optimizer=opt)
		# model.compile(loss='mse', optimizer=opt)
		model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
		model.save(model_file_name)
	
	else:
		from keras.models import load_model
		model = load_model(model_file_name, custom_objects={'weighted_mse': weighted_mse})

	y_hat_val = model.predict(val_X)
	y_hat_test = model.predict(test_X)

	# predict last
	last = model.predict(np.expand_dims(last_values, axis=0))
	if(calc_val_error):
		# for validation
		rmses_val = []
		rmse_val = 0
		#weigth = 1.0
		#step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(y_hat_val[:, i].ravel()), n_features))
			tmp[:, 0] = y_hat_val[:, i].ravel()
			y_hat_val[:, i] = scaler.inverse_transform(tmp)[:, 0]

			tmp = np.zeros((len(val_y[:, i].ravel()), n_features))
			tmp[:, 0] = val_y[:, i].ravel()
			val_y[:, i] = scaler.inverse_transform(tmp)[:, 0]

			rmses_val.append(math.sqrt(mean_squared_error(val_y[:, i], y_hat_val[:, i])))
			#rmse_val += rmses_val[-1]*weigth
			#weigth -= step
		rmse_val = np.mean(rmses_val)
	else:
		rmse_val = None

	if(calc_test_error):
		# for test
		rmses = []
		rmse = 0
		#weigth = 1.5
		#step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(y_hat_test[:, i].ravel()), n_features))
			tmp[:, 0] = y_hat_test[:, i].ravel()
			y_hat_test[:, i] = scaler.inverse_transform(tmp)[:, 0]

			tmp = np.zeros((len(test_y[:, i].ravel()), n_features))
			tmp[:, 0] = test_y[:, i].ravel()
			test_y[:, i] = scaler.inverse_transform(tmp)[:, 0]

			rmses.append(math.sqrt(mean_squared_error(test_y[:, i], y_hat_test[:, i])))
			#rmse += rmses[-1]*weigth
			#weigth -= step
		rmse = np.mean(rmses)
	else:
		rmse = None

	# transform last values
	tmp = np.zeros((last.shape[1], n_features))
	tmp[:, 0] = last
	last = scaler.inverse_transform(tmp)[:, 0]

	return rmse, rmse_val, test_y, y_hat_test, val_y, y_hat_val, last

############################ LSTM no slidding windows ################################
def model_lstm_noSliddingWindows(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_epochs, lr, n_features, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name):
	"""
		
		Función para crear, entrenar y calcular el error del modelo LSTM con el enfoque de no tener el concepto de ventana deslizante también sirve para predecir, se maneja un unico *batch* que tiene
		toda la secuencia y tampoco se maneja el concepto de *lags* por que los *lags* son toda la secuencia, además no se tien el concepto de estados escondidos ya que el número de estados 
		escondidos es igual al número de *features* y gracias a esto se eliminan parámetros lo que hace mas liviano el modelo y más fácil de optimizar

		Parámetros:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de validación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de validación
		- test_y -- Arreglo de numpy, observaciones de *testing*
		- n_series -- Entero, el número de time steps a predecir en el futuro
		- n_epochs -- Entero, número de epocas de entrenamiento
		- lr -- Flotante, tasa de entrenamiento del modelo
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- last_values -- Arreglo de numpy, últimos valores para realizar la predicción
		- calc_val_error -- Booleano, indica si se calcula el error de validación, si no se calcula se devuelve None en este campo
		- calc_test_error -- Booleano, indica si se calcula el error de *testing*, si no se calcula se devuelve None en este campo
		- verbosity -- Entero, nivel de verbosidad de la ejecución del modelo, entre más alto más información se mostrará el límite es 4 y debe ser mayor o igual a 0
		- only_predict -- Booleano, indica si solo se desea predecir o si también se desea entrenar el modelo. True sería solo predecir, False también entrenaría el modelo
		- model_file_name -- *String*, nombre del archivo donde se guardará y/o se cargará el modelo
		
		Retorna: 
		- rmse -- Flotante, raíz del error medio cuadrático de *testing*; retorna None si calc_test_error es False
		- rmse_val -- Flotante, raíz del error medio cuadrático de validación; retorna None si calc_val_error es False
		- obs_test -- Arreglo de numpy, observaciones de la particón de *testing* en escala real
		- preds_test -- Arreglo de numpy, predicciones de la partición de *testing* en escala real
		- obs_val -- Arreglo de numpy, observaciones de la particón de validación en escala real
		- preds_val -- Arreglo de numpy, predicciones de la partición de validación en escala real
		- last -- arreglo de numpy, predicción de los últimos valores

	"""
	n_out = n_series
	from keras.layers import LSTM, Input
	from keras.models import Model
	import time

	if(not only_predict):
		print('training...')
		from keras.models import Sequential
		from keras.optimizers import Adam

		drop_p = 0.05
		lr_decay = 0.0
		verbose_dict = {0:0, 1:2, 2:1}
		verbose = 0 if verbosity < 3 else min(verbosity - 2, 2)
		verbose = verbose_dict[verbose]

		train_model = Sequential()

		train_model.add(LSTM(train_X.shape[1], batch_input_shape=(1, None, train_X.shape[1]), return_sequences=True, stateful=True))

		opt = Adam(lr=lr, decay=lr_decay)
		train_model.compile(loss=weighted_mse, optimizer=opt)

		for epoch in range(n_epochs):
			train_model.fit(np.expand_dims(train_X, axis=0), np.expand_dims(train_y, axis=0), validation_data=(np.expand_dims(val_X, axis=0), np.expand_dims(val_y, axis=0)),epochs=1, verbose=verbose, shuffle=False, batch_size=1)
			train_model.reset_states()
		train_model.save(model_file_name)
	
	else:
		from keras.models import load_model
		train_model = load_model(model_file_name, custom_objects={'weighted_mse': weighted_mse})

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
		#weigth = 1.5
		#step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(preds_val[:, i].ravel()), n_features))
			tmp[:, 0] = preds_val[:, i].ravel()
			preds_val[:, i] = scaler.inverse_transform(tmp)[:, 0]

			tmp = np.zeros((len(obs_val[:, i].ravel()), n_features))
			tmp[:, 0] = obs_val[:, i].ravel()
			obs_val[:, i] = scaler.inverse_transform(tmp)[:, 0]

			rmses_val.append(math.sqrt(mean_squared_error(obs_val[:, i], preds_val[:, i])))
			#rmse_val += rmses_val[-1]*weigth
			#weigth -= step
		rmse_val = np.mean(rmses_val)
	else:
		rmse_val, preds_val, obs_val = None, None, None


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
		#weigth = 1.5
		#step = 0.1
		for i in range(n_out):
			tmp = np.zeros((len(preds_test[:, i].ravel()), n_features))
			tmp[:, 0] = preds_test[:, i].ravel()
			preds_test[:, i] = scaler.inverse_transform(tmp)[:, 0]

			tmp = np.zeros((len(obs_test[:, i].ravel()), n_features))
			tmp[:, 0] = obs_test[:, i].ravel()
			obs_test[:, i] = scaler.inverse_transform(tmp)[:, 0]

			rmses.append(math.sqrt(mean_squared_error(obs_test[:, i], preds_test[:, i])))
			#rmse += rmses[-1]*weigth
			#weigth -= step
		rmse = np.mean(rmses)
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

	return rmse, rmse_val, obs_test, preds_test, obs_val, preds_val, last.ravel()

###################### random forest ##########################
def model_random_forest(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_estimators, max_features, min_samples, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name):
	"""
		
		Función para crear, entrenar y calcular el error del modelo *random forest*, también sirve para predecir
		Este modelo se crea con la libreria sklearn con el algoritmo RandomForestregressor

		Parámetros:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de validación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de validación
		- test_y -- Arreglo de numpy, observaciones de *testing*
		- n_series -- Entero, el número de time steps a predecir en el futuro
		- n_estimators -- Entero, número de árboles que se generaran en el bosque
		- max_features -- Entero, número máximo de *features* que tendrá cada árbol dentro del bosque
		- min_samples -- Entero, número minimo de ejemplos encesarios para que la partición sea una hoja del árbol
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaran para entrenar
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- last_values -- Arreglo de numpy, últimos valores para realizar la predicción
		- calc_val_error -- Booleano, indica si se calcula el error de validación, si no se calcula se devuelve None en este campo
		- calc_test_error -- Booleano, indica si se calcula el error de *testing*, si no se calcula se devuelve None en este campo
		- verbosity -- Entero, nivel de verbosidad de la ejecución del modelo, entre más alto más información se mostrará el límite es 4 y debe ser mayor o igual a 0
		- only_predict -- Booleano, indica si solo se desea predecir o si también se desea entrenar el modelo. True sería solo predecir, False también entrenaría el modelo
		- model_file_name -- *String*, nombre del archivo donde se guardará y/o se cargará el modelo

		Retorna:
		- rmse -- Flotante, raíz del error medio cuadrático de *testing*; retorna None si calc_test_error es False
		- rmse_val -- Flotante, raíz del error medio cuadrático de validación; retorna None si calc_val_error es False
		- y -- Arreglo de numpy, observaciones de la particón de *testing* en escala real
		- y_hat -- Arreglo de numpy, predicciones de la partición de *testing* en escala real
		- y_valset -- Arreglo de numpy, observaciones de la particón de validación en escala real
		- y_hat_val -- Arreglo de numpy, predicciones de la partición de validación en escala real
		- last -- arreglo de numpy, predicción de los últimos valores

	"""
	from sklearn.externals import joblib
	if(not only_predict):
		print('training...')
		from sklearn.ensemble import RandomForestRegressor

		verbose = 0 if verbosity < 3 else min(verbosity - 2, 2)
		model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples, n_jobs=-1, verbose=verbose)
		model.fit(train_X, train_y.ravel())
		joblib.dump(model, model_file_name)
	else:
		model = joblib.load(model_file_name)
	if(calc_val_error):
		y_valset, y_hat_val, rmse_val = calculate_rmse(n_series, n_features, n_lags, val_X, val_y, scaler, model)
	else:
		rmse_val, y_valset, y_hat_val = None, None, None

	if(calc_test_error):
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
	else:
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
		rmse = None

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, rmse_val, y, y_hat, y_valset, y_hat_val, last

####################### ada boost ###############################
def model_ada_boost(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_estimators, lr, max_depth, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name):
	"""
		
		Función para crear, entrenar y calcular el error del modelo *ada boost*, también sirve para predecir
		Este modelo se construye con la libreria sklearn con el algoritmo AdaBoostRegressor y este ada boost se contruye a partir de árboles de decisión

		Parámetros:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de validación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de validación
		- test_y -- Arreglo de numpy, observaciones de *testing*
		- n_series -- Entero, el número de time steps a predecir en el futuro
		- n_estimators -- Entero, número de árboles que se generaran en el bosque
		- lr -- Flotante, tasa de entrenamiento del modelo
		- max_depth -- Entero, profundida máxima del árbol de decisión que se utiliza para construir el *ada boost*
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaran para entrenar
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- last_values -- Arreglo de numpy, últimos valores para realizar la predicción
		- calc_val_error -- Booleano, indica si se calcula el error de validación, si no se calcula se devuelve None en este campo
		- calc_test_error -- Booleano, indica si se calcula el error de *testing*, si no se calcula se devuelve None en este campo
		- verbosity -- Entero, nivel de verbosidad de la ejecución del modelo, entre más alto más información se mostrará el límite es 4 y debe ser mayor o igual a 0
		- only_predict -- Booleano, indica si solo se desea predecir o si también se desea entrenar el modelo. True sería solo predecir, False también entrenaría el modelo
		- model_file_name -- *String*, nombre del archivo donde se guardará y/o se cargará el modelo

		Retorna:
		- rmse -- Flotante, raíz del error medio cuadrático de *testing*; retorna None si calc_test_error es False
		- rmse_val -- Flotante, raíz del error medio cuadrático de validación; retorna None si calc_val_error es False
		- y -- Arreglo de numpy, observaciones de la particón de *testing* en escala real
		- y_hat -- Arreglo de numpy, predicciones de la partición de *testing* en escala real
		- y_valset -- Arreglo de numpy, observaciones de la particón de validación en escala real
		- y_hat_val -- Arreglo de numpy, predicciones de la partición de validación en escala real
		- last -- arreglo de numpy, predicción de los últimos valores

	"""
	from sklearn.externals import joblib
	if(not only_predict):
		print('training...')
		from sklearn.ensemble import AdaBoostRegressor
		from sklearn.tree import DecisionTreeRegressor

		model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators, learning_rate=lr)
		model.fit(train_X, train_y.ravel())
		joblib.dump(model, model_file_name)
	else:
		model = joblib.load(model_file_name)
	
	if(calc_val_error):
		y_valset, y_hat_val, rmse_val = calculate_rmse(n_series, n_features, n_lags, val_X, val_y, scaler, model)
	else:
		rmse_val, y_valset, y_hat_val = None, None, None

	if(calc_test_error):
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
	else:
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
		rmse = None
	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, rmse_val, y, y_hat, y_valset, y_hat_val, last

####################################### SVM ##############################
def model_svm(train_X, val_X, test_X, train_y, val_y, test_y, n_series, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name):
	"""
		
		Función para crear, entrenar y calcular el error del modelo SVM, también sirve para predecir
		Este modelo se construye con la libreria sklearn con el algoritmo SVR (*support vector regressor*) el kernel actual es polinomial de grado 1 por que dió mejores resultados
		con un dataset con el que probamos pero este se peude cambiar en el futuro

		Parámetros:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de validación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de validación
		- test_y -- Arreglo de numpy, observaciones de *testing*
		- n_series -- Entero, el número de time steps a predecir en el futuro
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaran para entrenar
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- last_values -- Arreglo de numpy, últimos valores para realizar la predicción
		- calc_val_error -- Booleano, indica si se calcula el error de validación, si no se calcula se devuelve None en este campo
		- calc_test_error -- Booleano, indica si se calcula el error de *testing*, si no se calcula se devuelve None en este campo
		- verbosity -- Entero, nivel de verbosidad de la ejecución del modelo, entre más alto más información se mostrará el límite es 4 y debe ser mayor o igual a 0
		- only_predict -- Booleano, indica si solo se desea predecir o si también se desea entrenar el modelo. True sería solo predecir, False también entrenaría el modelo
		- model_file_name -- *String*, nombre del archivo donde se guardará y/o se cargará el modelo

		Retorna:
		- rmse -- Flotante, raíz del error medio cuadrático de *testing*; retorna None si calc_test_error es False
		- rmse_val -- Flotante, raíz del error medio cuadrático de validación; retorna None si calc_val_error es False
		- y -- Arreglo de numpy, observaciones de la particón de *testing* en escala real
		- y_hat -- Arreglo de numpy, predicciones de la partición de *testing* en escala real
		- y_valset -- Arreglo de numpy, observaciones de la particón de validación en escala real
		- y_hat_val -- Arreglo de numpy, predicciones de la partición de validación en escala real
		- last -- arreglo de numpy, predicción de los últimos valores

	"""
	from sklearn.externals import joblib
	if(not only_predict):
		print('training...')
		from sklearn.svm import SVR

		verbose = 0 if verbosity < 3 else min(verbosity - 2, 2)
		model = SVR(kernel='poly', degree=1, gamma='scale', verbose=verbose)
		model.fit(train_X, train_y.ravel())
		joblib.dump(model, model_file_name)
	else:
		model = joblib.load(model_file_name)

	if(calc_val_error):
		y_valset, y_hat_val, rmse_val = calculate_rmse(n_series, n_features, n_lags, val_X, val_y, scaler, model)
	else:
		rmse_val, y_valset, y_hat_val = None, None, None

	if(calc_test_error):
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
	else:
		y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)
		rmse = None

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, rmse_val, y, y_hat, y_valset, y_hat_val, last

###################################### ARIMA #########################################
def model_arima(train_X, val_X, test_X, train_y, val_y, test_y, n_series, d, q, n_features, n_lags, scaler, last_values, calc_val_error, calc_test_error, verbosity, only_predict, model_file_name):
	"""
		
		Función para crear, entrenar y calcular el error del modelo ARIMA, también sirve para predecir
		Este modelo se contruye con la libreria statsmodels con el algoritmo SARIMAX por que proporciona mayor estabilidad y por que el problema contiene variables exogenas
		la implementación de la libreria es un poco enredada y la documentación no es la mejor, pero en python en cuanto a modelos ARIMA es de lo mejor que hay
		Cuando se introducen parámetros que no se pueden calcular matematicamente por el modelo arima se devuelven errores de 90000000000 en validación y *testing* y las demás
		variables se retornan en None

		Parámetros:
		- train_X -- Arreglo de numpy, datos de entrenamiento
		- val_X -- Arreglo de numpy, datos de validación
		- test_X -- Arreglo de numpy, datos de *testing*
		- train_y -- Arreglo de numpy, observaciones de entrenamiento
		- val_y -- Arreglo de numpy, observaciones de validación
		- test_y -- Arreglo de numpy, observaciones de *testing*
		- n_series -- Entero, el número de time steps a predecir en el futuro
		- d -- Entero, cantidad de diferenciaciones necesarias para que la serie a predecir sea estacionaria
		- q -- Entero, parámetro para el componente de media móvil del modelo
		- n_features -- Entero, el número de *features* con los que se entrena el modelo, también se conocen como variables exogenas
		- n_lags -- Entero, el número de *lags* que se usaran para entrenar
		- scaler -- Instancia de la clase MinMaxScaler de la libreria sklearn, sirve para escalar los datos y devolverlos a la escala original posteriormente
		- last_values -- Arreglo de numpy, últimos valores para realizar la predicción
		- calc_val_error -- Booleano, indica si se calcula el error de validación, si no se calcula se devuelve None en este campo
		- calc_test_error -- Booleano, indica si se calcula el error de *testing*, si no se calcula se devuelve None en este campo
		- verbosity -- Entero, nivel de verbosidad de la ejecución del modelo, entre más alto más información se mostrará el límite es 4 y debe ser mayor o igual a 0
		- only_predict -- Booleano, indica si solo se desea predecir o si también se desea entrenar el modelo. True sería solo predecir, False también entrenaría el modelo
		- model_file_name -- *String*, nombre del archivo donde se guardará y/o se cargará el modelo

		Retorna:
		- rmse -- Flotante, raíz del error medio cuadrático de *testing*; retorna None si calc_test_error es False
		- rmse_val -- Flotante, raíz del error medio cuadrático de validación; retorna None si calc_val_error es False
		- test_y -- Arreglo de numpy, observaciones de la particón de *testing* en escala real
		- y_hat -- Arreglo de numpy, predicciones de la partición de *testing* en escala real
		- val_y -- Arreglo de numpy, observaciones de la particón de validación en escala real
		- y_hat_val -- Arreglo de numpy, predicciones de la partición de validación en escala real
		- inv_yhat[-1] -- arreglo de numpy, predicción de los últimos valores

	"""
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	from statsmodels.tsa.statespace.mlemodel import MLEResults
	from numpy.linalg.linalg import LinAlgError
	import warnings

	y_hat = []
	y_hat_val = []
	try:
		if(not only_predict):
			print('training...')
			verbose = 0 if verbosity < 3 else verbosity - 2
			model = SARIMAX(train_X[:, 0], exog=train_X[:, 1:], order=(n_lags, d, q), enforce_invertibility=False, enforce_stationarity=False, dynamic=False)
			model_fit = model.fit(disp=verbose, iprint=verbose, maxiter=200, method='powell')
			model_fit.save(model_file_name)
		else:
			model_fit = MLEResults.load(model_file_name)
		
		final_endogs = np.append(val_X[:, 0], test_X[:, 0], axis=0)
		final_exogs = np.append(val_X[:, 1:], test_X[:, 1:], axis=0)
		diff_train_original_model = len(train_X) - model_fit.nobs
		if(diff_train_original_model > 0):
			final_endogs = np.insert(final_endogs, 0, train_X[-diff_train_original_model:, 0], axis=0)
			final_exogs = np.insert(final_exogs, 0, train_X[-diff_train_original_model:, 1:], axis=0)

		output = model_fit.predict(len(train_X), len(train_X) + len(val_X) + len(test_X) - 1, exog=final_exogs, endog=final_endogs)
		y_hat_val.extend(output[:len(val_y)])
		y_hat.extend(output[len(val_y):len(val_y)+len(test_y)])
		
		if(calc_val_error):
			tmp = np.zeros((len(y_hat_val), n_features))
			tmp[:, 0] = y_hat_val
			y_hat_val = scaler.inverse_transform(tmp)[:, 0]

			tmp = np.zeros((len(val_y), n_features))
			tmp[:, 0] = val_y
			val_y = scaler.inverse_transform(tmp)[:, 0]

			rmse_val = math.sqrt(mean_squared_error(val_y, y_hat_val))
		else:
			rmse_val, val_y, y_hat_val = None, None, None
		
		if(calc_test_error):
			tmp = np.zeros((len(y_hat), n_features))
			tmp[:, 0] = y_hat
			y_hat = scaler.inverse_transform(tmp)[:, 0]

			tmp = np.zeros((len(test_y), n_features))
			tmp[:, 0] = test_y
			test_y = scaler.inverse_transform(tmp)[:, 0]

			rmse = math.sqrt(mean_squared_error(test_y, y_hat))
		else:
			rmse, test_y, y_hat = None, None, None

		last = output[-1]
		last = last.reshape(-1, 1)
		Xs = np.ones((last.shape[0], n_lags * n_features))
		inv_yhat = np.concatenate((last, Xs[:, -(n_features - n_series):]), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, 0:n_series]
	except (ValueError, LinAlgError) as exc:
		print(exc)
		return 9e+10, 9e+10, None, None, None, None, None


	return rmse, rmse_val, test_y, y_hat, val_y, y_hat_val, inv_yhat[-1]
