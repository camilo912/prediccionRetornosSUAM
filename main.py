import numpy as np
import pandas as pd
import predictor
import csv
from matplotlib import pyplot as plt

def load_data(file_name, h, i):
	dataset = pd.read_csv(file_name, header=h, index_col=i)
	return dataset

def split_data(data, separator):
	train = data[:separator] 
	test = data[separator:]
	return train, test

def main():
	model = 5 # id of model to use
	parameters = 0 # Set to True for performing bayes optimization looking for best parameters
	select = 0 # set to True for performing feature selection
	original = 1 # set to True for training with original data (not feature selected)
	time_steps = 10
	max_vars = 25
	#file_name = 'forecast-competition-training.csv'
	input_file_name = 'data/forecast-competition-complete.csv'
	header, index_col = 0, 0
	dataframe = load_data(input_file_name, header, index_col)
	df = pd.DataFrame()
	output_file_name = 'results/salida_' + str(time_steps) + '_periodos.csv'
	f = open(output_file_name, 'w')
	writer = csv.writer(f)

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
	fin = 421
	step = 10
	for i in range(ini, fin, step): # 500 max for 10 time steps prediction maximun 490
		if(select and i > ini):
			select = 0
			original = 0
		print(i)
		train, test = split_data(dataframe.values, i)

		pred = predictor.predictor(train, model, parameters, select, original, time_steps, max_vars)#[0]

		actual = test[0:len(pred), 0]
		#print(pred.shape)

		print('pred:', pred)
		print('actual:', actual)
		df = df.append(pd.Series([pred]), ignore_index=True)
		writer.writerow([pred])
		#o.append(actual)
		p.append(pred)
		#o.extend(actual)
		#p.extend(pred)
		#plt.plot(pred, color='r')
		#plt.plot(actual, color='b')
		#plt.show()
	# plt.plot(p, color='r')
	# plt.plot(o, color='b', marker='*', linestyle='-.')
	datos = dataframe.values[ini:fin+20]
	plt.plot(datos[:, 0], marker='*', linestyle='-.')
	# for i in range(int(np.floor((fin - ini)/step))):
	for i in range(len(p)):
		pad = [None for j in range(i*step)]
		plt.plot(pad + list(p[i]))

	plt.show()

	plt.show()

	df.columns=[out]
	# print(df)
	f.close()


if __name__ == '__main__':
	main()