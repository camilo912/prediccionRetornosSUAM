import numpy as np
import pandas as pd
import predictor
import csv
from matplotlib import pyplot as plt

def split_data(data, separator):
	train = data[:separator] 
	test = data[separator:]
	return train, test

def main():

	#### Models ids ####
	# 0 -> LSTM with sliddding window
	# 1 -> Random Forest....................(only available for 1 time step)
	# 2 -> AdaBoost.........................(only available for 1 time step)
	# 3 -> SVM..............................(only available for 1 time step)
	# 4 -> Sarima...........................(only available for 1 time step)
	# 5 -> LSTM without slidding windows

	# Parameters
	model = 5 # id of model to use
	parameters = 0 # Set to True for performing bayes optimization looking for best parameters
	select = 0 # set to True for performing feature selection
	original = 1 # set to True for training with original data (not feature selected)
	time_steps = 10 # number of periods in the future to predict
	max_vars = 25 # maximum number of variables for taking in count for variable selection
	plots_level = 0 # level of log plots

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
	fin = 500 # 500 max for 10 time steps prediction maximun 490
	step = time_steps
	assert fin <= 500
	for i in range(ini, fin, step): 
		if(select and i > ini):
			select = 0
			original = 0
		print(i)
		train, test = split_data(dataframe.values, i)

		pred = predictor.predictor(train, model, parameters, select, original, time_steps, max_vars)

		actual = test[0:time_steps, 0]

		print('prediction:', pred)
		print('observation:', actual)
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
	plt.show()

	df.columns=[out]
	df.to_csv(output_file_name)


if __name__ == '__main__':
	main()