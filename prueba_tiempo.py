import pandas as pd
import numpy as np
import modelos
import utils
import csv
import math

from sklearn.metrics import mean_squared_error

from timeit import default_timer as timer

def transform_values(data, n_lags, n_series, dim):
	reframed = utils.series_to_supervised(data, n_lags, n_series)

	# wall = 200 - (n_lags - 1)
	wall = len(reframed) - 25

	values = reframed.values
	n_features = data.shape[1]
	n_obs = n_lags * n_features

	cols = ['var1(t)']
	cols += ['var1(t+%d)' % (i) for i in range(1, n_series)]
	y_o = reframed[cols].values

	train_X, train_y = values[:wall, :n_obs], y_o[:wall, -n_series:]
	test_X, test_y = values[wall:, :n_obs], y_o[wall:, -n_series:]

	if(dim):
		train_X = train_X.reshape((train_X.shape[0], n_lags, n_features))
		test_X = test_X.reshape((test_X.shape[0], n_lags, n_features))
	return train_X, test_X, train_y, test_y

def weighted_mse(yTrue, yPred):
	"""
		Función para calcular el error personalizado del modelo LSTM, este error personalizado es la raíz del error cuadrático medio dandole más importancia a los primeros *lags*

		Parámetros:
		- yTrue -- valores con los cuales e va a comaprar las predicciones
		- yPred -- predicciones para calcular el error

		Retorna:
		- [valor] -- error cuadrático medio ponderado
	"""
	from keras import backend as K
	ones = K.ones_like(yTrue[0,:]) # a simple vector with ones shaped as (n_series,)
	idx = K.cumsum(ones) # similar to a 'range(1,n_series+1)'

	return K.mean((1/idx)*K.square(yTrue-yPred))

def train_model(train_X, test_X, train_y, test_y, n_series, n_epochs, batch_size, n_hidden, n_features, n_lags, scaler, verbosity, n_rnn, n_dense, activation, drop_p, returns):
	n_out = n_series

	from keras.layers import Dense, Dropout, LSTM, RepeatVector, Reshape
	from keras.models import Sequential
	from keras.optimizers import Adam
	from keras import backend as K

	verbose_dict = {0:0, 1:2, 2:1}
	activation_dict = {0:'tanh', 1:'relu'}
	verbose = 0 if verbosity < 3 else min(verbosity - 2, 2)
	verbose = verbose_dict[verbose]

	K.clear_session()
	model = Sequential()
	if(n_series == 1):
		for i in range(n_rnn):
			model.add(LSTM(n_hidden, return_sequences=True, activation=activation_dict[activation]))
			model.add(Dropout(drop_p))
		model.add(LSTM(n_hidden, activation=activation_dict[activation]))
		for i in range(n_dense):
			model.add(Dense(n_hidden))
		model.add(Dense(n_out))
	else:
		model.add(LSTM(n_hidden, input_shape=(n_lags, n_features), return_sequences=False, activation=activation_dict[activation]))
		model.add(RepeatVector(n_out))
		for i in range(n_rnn):
			model.add(LSTM(n_hidden, return_sequences=True, activation=activation_dict[activation]))
			model.add(Dropout(drop_p))
		for i in range(n_dense):
			model.add(Dense(n_hidden))	
		model.add(Dense(1))
		model.add(Reshape((n_out,)))

	opt = Adam(lr=0.001)#, clipvalue=0.005, decay=0.005)
	model.compile(loss=weighted_mse, optimizer=opt)
	history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, verbose=verbose, shuffle=False)
	
	y_hat = model.predict(test_X)
	if(type(y_hat) == list): print(y_hat)

	rmses = []
	rmse = 0
	for i in range(n_out):
		tmp = np.zeros((len(y_hat[:, i].ravel()), n_features))
		tmp[:, 0] = y_hat[:, i].ravel()
		y_hat[:, i] = scaler.inverse_transform(tmp)[:, 0]

		tmp = np.zeros((len(test_y[:, i].ravel()), n_features))
		tmp[:, 0] = test_y[:, i].ravel()
		test_y[:, i] = scaler.inverse_transform(tmp)[:, 0]

		rmses.append(math.sqrt(mean_squared_error(test_y[:, i], y_hat[:, i])))
	rmse = np.mean(rmses)

	if(returns):
		
		dir_acc = utils.get_returns_direction_accuracy(test_y.ravel(), y_hat.ravel())
	else:
		dir_acc = utils.get_direction_accuracy(test_y.ravel(), y_hat.ravel())

	return rmse, test_y, y_hat, dir_acc, model

def objective(params, id_model, values, scaler, n_features, n_series, verbosity, model_file_name, MAX_EVALS, returns):
	global ITERATION

	ITERATION += 1
	out_file = 'trials/gbm_trials_' + ID_TO_MODELNAME[id_model] + '_' + str(MAX_EVALS) + '.csv'
	# print(ITERATION, params)

	calc_val_error = True
	calc_test_error = False
	if(id_model == 0):
		if(model_file_name == None): model_file_name = 'models/trials-lstm.h5'

		# Make sure parameters that need to be integers are integers
		for parameter_name in ['n_lags', 'n_epochs', 'batch_size', 'n_hidden', 'n_dense', 'n_rnn', 'activation']:
			params[parameter_name] = int(params[parameter_name])

		start = timer()
		train_X, test_X, train_y, test_y = transform_values(values, params['n_lags'], n_series, 1)

		rmse, test_y, _, dir_acc, _ = train_model(train_X, test_X, train_y, test_y, n_series, params['n_epochs'], params['batch_size'], params['n_hidden'], n_features, 
														params['n_lags'], scaler, verbosity, params['n_rnn'], params['n_dense'], params['activation'], params['drop_p'], returns)

		if(not returns):
			mean = np.mean(np.abs(test_y))
			rmse = rmse/mean + (1 - dir_acc)
		run_time = timer() - start

	# Write to the csv file
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([rmse, params, ITERATION, run_time])
	of_connection.close()

	return -1 * rmse

def bayes_optimization(id_model, MAX_EVALS, values, scaler, n_features, n_series, original, verbosity, model_file_name, returns):
	from bayes_opt import BayesianOptimization

	global ITERATION
	ITERATION = 0

	if(id_model == 0):
		if(n_series == 1):
			space = {'activation': (0.1, 1.9),
					'batch_size': (20, 100),
					'drop_p': (0, 0.75),
					'n_dense': (0, 3),
					'n_epochs': (10, 200),
					'n_hidden': (5, 300),
					'n_lags': (1, (min(30, int(len(values)/2)))),
					'n_rnn': (0, 3)}
		elif(n_series > 1):
			space = {'activation': (0.1, 1.9),
					'batch_size': (20, 100),
					'drop_p': (0, 0.3),
					'n_dense': (0, 3),
					'n_epochs': (250, 500),
					'n_hidden': (50, 300),
					'n_lags': (2, (min(30, int(len(values)/2)))),
					'n_rnn': (1, 3)}
		func = lambda activation, batch_size, drop_p, n_dense, n_epochs, n_hidden, n_lags, n_rnn: objective({'activation':activation , 'batch_size':batch_size, 'drop_p':drop_p, 'n_dense':n_dense, 'n_epochs':n_epochs, 'n_hidden':n_hidden, 'n_lags':n_lags, 'n_rnn':n_rnn}, id_model, values, scaler, n_features, n_series, verbosity, model_file_name, MAX_EVALS, returns)

	# File to save results
	out_file = 'trials/gbm_trials_' + ID_TO_MODELNAME[id_model] + '_' + str(MAX_EVALS) + '.csv'
	of_connection = open(out_file, 'w')
	writer = csv.writer(of_connection)

	# Write the headers to the file
	writer.writerow(['id_model: ' + str(id_model), 'original: ' + str(original), 'returns: ' + str(returns)])
	writer.writerow(['rmse', 'params', 'iteration', 'train_time'])
	of_connection.close()

	optimizer = BayesianOptimization(f=func, pbounds=space, verbose=2, random_state=np.random.randint(np.random.randint(100)))
	if(id_model == 0):
		optimizer.probe(params={'activation':1.0, 'batch_size':10.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':300.0, 'n_hidden':50.0, 'n_lags':10.0, 'n_rnn':0.0})
		optimizer.probe(params={'activation':1.0, 'batch_size':81.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':200.0, 'n_hidden':250.0, 'n_lags':15.0, 'n_rnn':0.0})
		optimizer.probe(params={'activation':1.0, 'batch_size':81.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':200.0, 'n_hidden':269.0, 'n_lags':9.0, 'n_rnn':0.0})
		# optimizer.probe(params={'activation':1.0, 'batch_size':81.0, 'drop_p':0.0, 'n_dense':0.0, 'n_epochs':200.0, 'n_hidden':269.0, 'n_lags':25.0, 'n_rnn':0.0})
	optimizer.maximize(init_points=10, n_iter=MAX_EVALS, acq='ucb', kappa=5, alpha=1e-3)


	# store best results
	best = optimizer.max['params']
	of_connection = open('trials/bests.txt', 'a')
	writer = csv.writer(of_connection)
	if(id_model == 0):
		writer.writerow([optimizer.max['target'], best['activation'], best['batch_size'], best['drop_p'], best['n_dense'], best['n_epochs'], best['n_hidden'], best['n_lags'], best['n_rnn'], MAX_EVALS])

	of_connection.close()

	return best



ID_TO_MODELNAME = {0:'lstm', 1:'randomForest', 2:'adaBoost', 3:'svm', 4:'arima', 5:'lstmNoSW'}

df_test = pd.read_csv('data/data_16_11_2018_differentiated.csv', header=0, index_col=0)
df = pd.read_csv('data/data_selected.csv', header=0, index_col=0)
selected_features = list(df.columns)

values, scaler = utils.normalize_data(df_test[selected_features].values, scale=(-1, 1))
n_features = values.shape[1]

# parametros iniciales
id_model = 0
MAX_EVALS = 50
time_steps = 10
original = False
verbosity = 0
model_file_name = None
returns = True
calc_val_error = True


best = bayes_optimization(id_model, MAX_EVALS, values, scaler, n_features, time_steps, original, verbosity, model_file_name, returns)

if(model_file_name == None): model_file_name = 'models/lstm_%dtimesteps.h5' % (time_steps)
n_lags, n_epochs, batch_size, n_hidden = int(best['n_lags']), int(best['n_epochs']), int(best['batch_size']), int(best['n_hidden'])
n_rnn, n_dense, activation, drop_p = int(best['n_rnn']), int(best['n_dense']), int(best['activation']), best['drop_p']
f = open('parameters/optimized_lstm_%dtimesteps.pars' % time_steps, 'w')
f.write('%d, %d, %d, %d\n' % (n_lags, n_epochs, batch_size, n_hidden))
f.close()
train_X, test_X, train_y, test_y = transform_values(values, n_lags, time_steps, 1)

rmse, y, y_hat, dir_acc, model = train_model(train_X, test_X, train_y, test_y, time_steps, n_epochs, batch_size, n_hidden, n_features, n_lags, 
												scaler, verbosity, n_rnn, n_dense, activation, drop_p)

print('rmse: %s ' % rmse)

print('direction accuracy: %f%%' % (dir_acc*100))

if(time_steps > 1):
	utils.plot_data_lagged_blocks([y[:, 0].ravel(), y_hat], ['y', 'y_hat'], 'Test plot')
else:
	utils.plot_data([y, y_hat], ['y', 'y_hat'], 'Test plot')