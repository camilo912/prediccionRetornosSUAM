import numpy as np
import pandas as pd
import series_predictor
import csv
from matplotlib import pyplot as plt

def split_data(data, separator):
	train = data[:separator] 
	test = data[separator:]
	return train, test

def main():
	"""
		Main del proyecto, este archivo no es necesario, puede ser personalizado o incluso crear uno nuevo, este es simplemente el estilo del creador de este proyecto

		contien la interacción de un usuario final con el proyecto mediante este archivo se leen los datos de un archivo .csv, se inicializan los hyperparámetros para la ejecución
		se invoca al proyecto para realizar las predicciones y se grafican los resultados


		#### Models ids ####
		# 0 -> LSTM with sliddding window
		# 1 -> Random Forest....................(only available for 1 time step)
		# 2 -> AdaBoost.........................(only available for 1 time step)
		# 3 -> SVM..............................(only available for 1 time step)
		# 4 -> Arima...........................(only available for 1 time step)
		# 5 -> LSTM without slidding windows

		Se manjean 11 hyperparámetros principales
		- model -- Entero, el id del modelo que se va a utilizar (ver ids arriba)
		- parameters -- Booleano, indica si se va a hacer optimzación de parámetros
		- select -- Booleano, indica si se va a hacer selección de variables
		- original -- Booleano, indica si se va a entrenar con las variables originales o con las variables seleccionadas, True es con las originales (si select es True este es False automaticamente)
		- time_steps -- Entero, número de tiempos en el futuro que se van a predecir
		- max_vars -- Entero, número máximo de variables a ser seleccionadas por el algoritmo de selección de variables, si select es False, este parámetro no importa
		- verbosity -- Entero, número que indica el nivel de verbosidad de la ejecución, hasta 2 muestra gráficas, de ahí para arriba muestra *logs* de la ejecución
		- parameters_file_name -- String, nombre del archivo donde se va a guardar o cargar el modelo entrenado, si este parámetro es None, se utilzian los parámetros por *default*
		- MAX_EVALS -- Entero, número de iteraciones de la optimización bayesiana, si parameters es False este parámetro no importa
		- only_predict -- Booleano, indica si solo se va a predecir o tambien se va a entrenar el modelo
		- model_file_name -- String, nombre del archivo donde se va a guardar y/o cargar el modelo entrenado, si only_predict es False este parámetro no importa

		Hay 1 hyperparámetro adicional que es:
		- predicting -- Entero, id de la columna dentro del dataframe de la variable objetivo a predecir, para que en el caso que esta no sea la posición 0, se intercambie

	"""

	# Parameters
	model = 1 # id of model to use
	parameters = 1 # Set to True for performing bayes optimization looking for best parameters
	select = 1 # set to True for performing feature selection
	original = 0 # set to True for training with original data (not feature selected)
	time_steps = 1 # number of periods in the future to predict
	max_vars = 50 # maximum number of variables for taking in count for variable selection
	verbosity = 0 # level of logs
	parameters_file_name = None #'parameters/default_lstm_%dtimesteps.pars' % time_steps
	MAX_EVALS = 100
	only_predict = False
	model_file_name = None # 'models/lstm-noSW-prueba.h5'

	input_file_name = 'data/forecast-competition-complete.csv'
	dataframe = pd.read_csv(input_file_name, header=0, index_col=0)
	df = pd.DataFrame()
	output_file_name = 'results/salida_' + str(time_steps) + '_periodos.csv'

	# chose the feature to predict
	predicting = 0
	cols = dataframe.columns
	predicting = cols[predicting]
	cols = set(cols)
	cols.remove(predicting)
	cols = [predicting] + list(cols)
	dataframe = dataframe[cols]

	out = dataframe.columns[0]
	mini = min(dataframe.loc[:,out].values)
	maxi = max(dataframe.loc[:,out].values)
	rango = maxi - mini

	o, p = [], []
	ini = 400
	fin = 410
	step = time_steps
	assert fin <= 500

	predictor = series_predictor.Predictor(model, original, time_steps)

	for i in range(ini, fin, step):
		# only select variables once 
		if(select and i > ini):
			select = 0
			original = 0
		
		# only optimize parameters once
		if(parameters and i > ini):
			parameters = 0
		
		print(i)
		train, test = split_data(dataframe.values, i)

		pred = predictor.predict(train, cols, parameters, select, max_vars, verbosity, parameters_file_name, MAX_EVALS, only_predict, model_file_name)

		actual = test[0:time_steps, 0]

		print('prediction:', pred)
		print('observation:', actual, end='\n\n')
		df = df.append(pd.Series([pred]), ignore_index=True)
		p.append(pred)
	datos = dataframe.values[ini:fin+step*2]
	plt.plot(datos[:, 0], marker='*', linestyle='-.', label='observations')
	if(time_steps > 1):
		for i in range(len(p)):
			pad = [None for j in range(i*step)]
			plt.plot(pad + list(p[i]))
	else:
		plt.plot(p, label='predictions')
		plt.legend()
	plt.suptitle('Predictions vs Observations', fontsize=16)
	plt.show()

	df.columns=[out]
	df.to_csv(output_file_name)


if __name__ == '__main__':
	main()