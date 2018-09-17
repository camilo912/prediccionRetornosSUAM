import numpy as np
import pandas as pd
import predictor
import csv

def load_data(file_name, h, i):
	dataset = pd.read_csv(file_name, header=h, index_col=i)
	return dataset

def split_data(data, separator):
	train = data[:separator] 
	test = data[separator:]
	return train, test

def main():
	model = 3
	parameters = 0
	#file_name = 'forecast-competition-training.csv'
	file_name = 'forecast-competition-complete.csv'
	header, index_col = 0, 0
	dataframe = load_data(file_name, header, index_col)
	df = pd.DataFrame()
	f = open('resultados.csv', 'w')
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

	for i in range(400, 401): # 500 max
		print(i)
		train, test = split_data(dataframe.values, i)

		pred = predictor.predictor(train, model, parameters)

		pred = pred[0]
		actual = test[0, 0]

		print('pred: %.20f' % pred)
		print('actual: %.20f' % actual)
		print('diff: %.20f, %.2f%%  \n\n' % (np.abs(actual - pred), np.abs((actual - pred) / rango)*100))
		df = df.append(pd.Series(pred), ignore_index=True)
		writer.writerow([pred])

	df.columns=[out]
	print(df)
	f.close()


if __name__ == '__main__':
	main()