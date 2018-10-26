import numpy as np
import pandas as pd
import utils

from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('data/forecast-competition-complete.csv', header=0, index_col=0)
values, scaler = utils.normalize_data(df.values)

wall = int(len(values)*0.6)
wall_val= int(len(values)*0.2)
train_X, val_X, test_X, last_values = values[:wall, :], values[wall:wall+wall_val,:], values[wall+wall_val:-1,:], values[-1,:]
train_y, val_y, test_y = values[1:wall+1,0], values[wall+1:wall+wall_val+1,0], values[wall+wall_val+1:,0]

d = 0
verbose=0
goods=[]

for n_lags in range(2):
	print(n_lags)
	for q in range(2):
		print(q)
		try:
			model = SARIMAX(train_X[:, 0], exog=train_X[:, 1:], order=(n_lags, d, q), enforce_invertibility=False, enforce_stationarity=False)
			model_fit = model.fit(disp=verbose, iprint=verbose, maxiter=200, method='powell')
			goods.append([n_lags, q])
		except:
			print('error')
			continue

print(goods)