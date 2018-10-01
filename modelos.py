from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

import numpy as np
import math

def calculate_rmse(n_series, n_features, n_lags, X, y, scaler, model):
	yhat = model.predict(X)
	Xs = np.ones((X.shape[0], n_lags * n_features))
	yhat = yhat.reshape(-1, n_series)
	#print(yhat[:, 0].reshape(-1, 1).shape)
	#print(Xs[:, -(n_features - 1):].shape)
	#print(Xs[:, -(n_features - n_series):].shape)
	inv_yhats = []
	for i in range(n_series):
		inv_yhat = np.concatenate((Xs[:, -(n_features - 1):], yhat[:, i].reshape(-1, 1)), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, -1]
		inv_yhats.append(inv_yhat)

	inv_yhat = np.array(inv_yhats).T
	
	# invert scaling for actual
	y = y.reshape((len(y), n_series))
	inv_ys = []
	for i in range(n_series):
		inv_y = np.concatenate((Xs[:, -(n_features - 1):], y[:, i].reshape(-1, 1)), axis=1)
		inv_y = scaler.inverse_transform(inv_y)
		inv_y = inv_y[:,-1]
		inv_ys.append(inv_y)

	inv_y = np.array(inv_ys).T


	# calculate RMSE	
	rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat)) / np.max(inv_y) * 100
	return inv_y, inv_yhat, rmse

def predict_last(n_series, n_features, n_lags, X, scaler, model, dim):
	if(dim):
		X = np.expand_dims(X, axis=0)
		# X = X.reshape(1, n_lags, -1) # only for dim, only for LSTM or RNN
	yhat = model.predict(X)
	if(not dim):
		yhat = yhat.reshape(-1, 1) # only for no dim
	Xs = np.ones((X.shape[0], n_lags * n_features))
	# # inv_yhat = np.concatenate((yhat, Xs[:, -(n_features - n_series):]), axis=1)
	# inv_yhat = np.concatenate((yhat, Xs[:, -(n_features - n_series):]), axis=1)
	# inv_yhat = scaler.inverse_transform(inv_yhat)

	# inv_yhat = inv_yhat[:,0:n_series]
	inv_yhats = []
	for i in range(n_series):
		inv_yhat = np.concatenate((Xs[:, -(n_features - 1):], yhat[:, i].reshape(-1, 1)), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)

		inv_yhat = inv_yhat[:, -1]
		inv_yhats.append(inv_yhat)

	inv_yhat = np.array(inv_yhats).T
	#return inv_yhat[-1]
	return inv_yhat

############################# LSTM ####################################
class Data_manager(Dataset):
	def __init__(self, X, y):
		self.X, self.y = X, y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		x = self.X[idx]
		y = self.y[idx]
		return x, y


def model_lstm(train_X, test_X, train_y, test_y, n_series, n_epochs, batch_size, lr, n_hidden, n_features, n_lags, scaler, last_values):
	# insert lag index to data
	train_X = np.insert(train_X, train_X.shape[-1], np.arange(train_X.shape[1])-n_lags, axis=2)
	test_X = np.insert(test_X, test_X.shape[-1], np.arange(test_X.shape[1])-n_lags, axis=2)
	n_features += 1

	trdm = Data_manager(train_X, train_y)
	tedm = Data_manager(test_X, test_y)

	if(len(train_X) % batch_size == 1 or len(test_X) % batch_size == 1):
		batch_size += 1

	trdl = DataLoader(trdm, batch_size=batch_size)
	tedl = DataLoader(tedm, batch_size=len(train_X))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Model(n_hidden, train_X.shape[2], 1, n_lags, device)
	opt = optim.Adam(model.parameters(), lr)
	loss_fnc = nn.MSELoss()
	historic_loss = []
	historic_train_loss = []
	debugs = []
	for epoch in range(n_epochs):
		train_loss = 0
		for X, y in trdl:
			X.to(device)
			y.to(device)

			out = []
			total_loss = 0
			for i in range(n_series):
				pred, ming1, ming2, avgg1, avgg2, maxg1, maxg2 = model(X)
				debugs.append([ming1, ming2, avgg1, avgg2, maxg1, maxg2])
				loss = loss_fnc(pred.view(-1), y[:, i])
				opt.zero_grad()
				loss.backward()
				opt.step()
				out.append(pred.detach().numpy().ravel())

				# agregar prediccion a la entrada
				X = cambiar_X(X, pred, device)

			out = np.array(out).T
			out = torch.Tensor(out, device=device)
			loss = loss_fnc(out, y)
			
			train_loss += loss.data.item()			

		test_loss = 0
		for X, y in tedl:
			with torch.no_grad():
				X.to(device)
				y.to(device)

				pred, loss = predict_recursively(X, y, model, loss_fnc, opt, n_series, device, False)
				test_loss += loss

		print('epoch: ', epoch)
		print('train loss: ', train_loss)
		print('test loss: ', test_loss, end='\n\n')
		historic_train_loss.append(train_loss)
		historic_loss.append(test_loss)
		# raise Exception('una iteracion exception')
	tmp = np.array(debugs)
	plt.subplot(3, 1, 1)
	plt.plot(tmp[:, 0], color='b')
	plt.plot(tmp[:, 1], color='r')
	plt.subplot(3, 1, 2)
	plt.plot(tmp[:, 2], color='b')
	plt.plot(tmp[:, 3], color='r')
	plt.subplot(3, 1, 3)
	plt.plot(tmp[:, 4], color='b')
	plt.plot(tmp[:, 5], color='r')
	plt.show()

	last, _ = predict_recursively(torch.Tensor(np.expand_dims(np.insert(last_values, last_values.shape[1], 0, axis=1), axis=0), device=device), torch.zeros(1, n_series).to(device), model, loss_fnc, opt, n_series, device, False)
	plt.plot(historic_train_loss, color='g', label='historic train loss')
	plt.plot(historic_loss, color='r', label='historic test loss')
	plt.show()

	return test_loss, y.detach().numpy(), pred.detach().numpy(), last.detach().numpy()

def predict_recursively(X, y, model, loss_fnc, opt, n_out, device, train):
	out = []
	total_loss = 0
	for i in range(n_out):
		pred, _, _, _, _, _, _ = model(X)
		if(train):
			loss = loss_fnc(pred.view(-1), y[:, i])
			opt.zero_grad()
			loss.backward()
			opt.step()
		out.append(pred.detach().numpy().ravel())

		# agregar prediccion a la entrada
		X = cambiar_X(X, pred, device)

	out = np.array(out).T
	out = torch.Tensor(out, device=device)
	loss = loss_fnc(out, y)
	
	total_loss += loss.data.item()
	
	return out, total_loss

		

def cambiar_X(X, pred, device):
	X = X.detach().numpy()
	X = np.roll(X, -1, 1) # deslizar el arreglo para que la ultima posicion sea la primera y la podamos eliminar
	X[:, -1, 1:] = X[:, -2, 1:] # colocar constantes las variables exogenas en el futuro
	X[:, -1, 0] = pred.detach().numpy().ravel() # agregar la prediccion para entrenar con ella
	X[:, -1, -1] = X[:, -1, -1] + 1.0
	X = torch.Tensor(X, device=device)
	return X

class Model(nn.Module):
	def __init__(self, n_hidden, n_in, n_out, n_lags, device):
		super().__init__()
		self.n_hidden, self.n_in, self.n_out, self.n_lags, self.device = n_hidden, n_in, n_out, n_lags, device
		
		# layers	
		self.activation = nn.ReLU().to(self.device)
		self.activation2 = nn.Tanh().to(self.device)
		self.rnn = nn.LSTM(self.n_in, self.n_hidden, batch_first=True)
		self.reduction = nn.Linear(self.n_hidden, 1)
		self.outer = nn.Linear(self.n_lags, n_out)
		
		# xavier initialization
		torch.nn.init.xavier_uniform_(self.reduction.weight)
		torch.nn.init.xavier_uniform_(self.outer.weight)
		torch.nn.init.xavier_uniform_(self.rnn.all_weights[0][0])
		torch.nn.init.xavier_uniform_(self.rnn.all_weights[0][1])

		# import pandas as pd
		# # load parameters
		# df = pd.read_csv('datos_saved_0.csv')
		# data = df.values
		# self.rnn.all_weights[0][0] = data[:self.rnn.all_weights[0][0].shape[1], :].T
		# self.rnn.all_weights[0][1] = data[self.rnn.all_weights[0][0].shape[1]:, :].T	

		# # save parameters
		# print(self.rnn.all_weights[0][0].shape)
		# print(self.rnn.all_weights[0][1].shape)
		# print(self.reduction.weight.shape)
		# print(self.outer.weight.shape)

		# a = self.rnn.all_weights[0][0].detach().numpy().T
		# b = self.rnn.all_weights[0][0].detach().numpy().T
		# c = np.append(a,b, axis=0)
		# d = pd.DataFrame(c)
		# d.to_csv('datos.csv')

		# # save lienar layers parameters
		# a = self.reduction.weight.detach().numpy().T
		# b = self.outer.weight.detach().numpy().T
		# c = np.append(a, b, axis=0)
		# d = pd.DataFrame(c)
		# d.to_csv('datos_linear.csv')

	def forward(self, seq):
		self.init_hidden(seq.size(0))
		# if(seq.size(0) > 1):
		# 	self.bn = nn.BatchNorm1d(seq.size(1))
		# 	seq = self.bn(seq)
		rnn_out, self.h = self.rnn(seq, self.h)
		# if(seq.size(0) > 1):
		# 	self.bn = nn.BatchNorm1d(rnn_out.size(1))
		# 	rnn_out = self.bn(rnn_out)
		out = self.reduction(rnn_out).view(rnn_out.size(0), rnn_out.size(1))
		# out = self.activation(out)
		# out = self.outer(out).view(rnn_out.size(0))
		out = out[:, -1].view(rnn_out.size(0))
		out = self.activation(out)
		grad1 = [abs(x) for x in self.rnn.all_weights[0][0].detach().numpy()]
		grad2 = [abs(x) for x in self.rnn.all_weights[0][1].detach().numpy()]
		grad3 = [abs(x) for x in self.reduction.weight.detach().numpy()]
		grad4 = [abs(x) for x in self.outer.weight.detach().numpy()]
		# suma = np.sum(grad1) + np.sum(grad2) + np.sum(grad3) + np.sum(grad4)
		# print('suma gradientes: ', suma)

		# print('min grad1: ', np.min(grad1))
		# print('min grad2: ', np.min(grad2))
		# print('min grad3: ', np.min(grad3))
		# print('min grad4: ', np.min(grad4))

		# print('avg grad1: ', np.mean(grad1))
		# print('avg grad2: ', np.mean(grad2))
		# print('avg grad3: ', np.mean(grad3))
		# print('avg grad4: ', np.mean(grad4))

		# print('max grad1: ', np.max(grad1))
		# print('max grad2: ', np.max(grad2))
		# print('max grad3: ', np.max(grad3))
		# print('max grad4: ', np.max(grad4))

		return out, np.min(grad1), np.min(grad2), np.mean(grad1), np.mean(grad2), np.max(grad1), np.max(grad2)

	def init_hidden(self, batch_size):
		self.h = (torch.zeros(1, batch_size, self.n_hidden), torch.zeros(1, batch_size, self.n_hidden))

###################### random forest ##########################
def model_random_forest(train_X, test_X, train_y, test_y, n_series, n_estimators, max_features, min_samples, n_features, n_lags, scaler, last_values):
	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples, n_jobs=4)
	model.fit(train_X, train_y.ravel())

	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, y, y_hat, last

####################### ada boost ###############################
def model_ada_boost(train_X, test_X, train_y, test_y, n_series, n_estimators, lr, n_features, n_lags, scaler, last_values):
	from sklearn.ensemble import AdaBoostRegressor
	model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=lr)
	model.fit(train_X, train_y.ravel())

	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, y, y_hat, last

####################################### SVM ##############################
def model_svm(train_X, test_X, train_y, test_y, n_series, n_features, n_lags, scaler, last_values):
	from sklearn.svm import SVR
	model = SVR(kernel='poly', degree=1)
	model.fit(train_X, train_y.ravel())

	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 0)

	return rmse, y, y_hat, last

###################################### ARIMA #########################################
def model_arima(train_X, test_X, train_y, test_y, n_series, d, q, n_features, n_lags, scaler, last_values):
	#from statsmodels.tsa.arima_model import ARIMA
	from statsmodels.tsa.statespace.sarimax import SARIMAX
	import warnings
	test_size = len(test_y)
	test_y = np.append(test_y, last_values)
	y_hat = []
	for i in range(test_size + 1):
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=ConvergenceWarning)
			model = SARIMAX(train_X, order=(n_lags, d, q))
			model_fit = model.fit(disp=0, maxiter=200, method='powell')
			output = model_fit.forecast()[0]
			y_hat.append(output)
			train_X = np.append(train_X, test_y[i])
	#print(test_y)
	#print(y_hat)
	rmse = math.sqrt(mean_squared_error(test_y, y_hat))
	model = SARIMAX(train_X, order=(n_lags, d, q))
	model_fit = model.fit(disp=0)
	last = model_fit.forecast()[0]
	last = last.reshape(-1, 1)
	Xs = np.ones((last.shape[0], n_lags * n_features))
	# inv_yhat = np.concatenate((yhat, Xs[:, -(n_features - n_series):]), axis=1)
	inv_yhat = np.concatenate((last, Xs[:, -(n_features - n_series):]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)

	inv_yhat = inv_yhat[:,0:n_series]

	return rmse, test_y, y_hat, inv_yhat[-1]