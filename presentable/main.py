import numpy as np
import pandas as pd
import predictor
from matplotlib import pyplot as plt


def split_data(data, separator):
	train = data[:separator] 
	test = data[separator:]
	return train, test

def main():
	model = 0 # id of model to use
	parameters = 0 # Set to True for performing bayes optimization looking for best parameters
	select = 0 # set to True for performing feature selection
	original = 1 # set to True for training with original data (not feature selected)
	file_name = 'data/forecast-competition-complete.csv'
	header, index_col = 0, 0
	dataframe = pd.read_csv(file_name, header=header, index_col=index_col)
	df = pd.DataFrame()
	results_file_name = 'results/salida_10_periodos.csv'

	# chose the feature to predict
	predicting = 0 # index of column to be predicted
	cols = dataframe.columns
	predicting = cols[predicting]
	cols = set(cols)
	cols.remove(predicting)
	cols = [predicting] + list(cols)
	dataframe = dataframe[cols]

	out = dataframe.columns[0]

	p = []
	ini = 400
	fin = 500
	step = 1

	for i in range(ini, fin, step): # 500 max for 10 time steps prediction maximun 490
		print(i)
		train, test = split_data(dataframe.values, i)

		pred = predictor.predictor(train, model, parameters, select, original)

		actual = test[0:len(pred), 0]

		print('pred:', pred)
		print('actual:', actual)
		df = df.append(pd.Series([pred]), ignore_index=True)

		p.append(pred)

	datos = dataframe.values[ini:fin+20]
	plt.plot(datos[:, 0], marker='*', linestyle='-.')
	for i in range(int(np.floor((fin - ini)/step))):
		pad = [None for j in range(i*step)]
		plt.plot(pad + list(p[i]))

	plt.show()

	df.columns=[out]
	df.to_csv(results_file_name)


if __name__ == '__main__':
	main()