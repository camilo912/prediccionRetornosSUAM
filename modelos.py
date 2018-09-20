from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# def model_lstm(train_X, test_X, train_y, test_y, n_series, n_a, n_epochs, batch_size, n_features, n_lags, scaler, last_values):
# 	from keras.models import Sequential
# 	from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
# 	# design network
# 	model = Sequential()
# 	model.add(LSTM(n_a, input_shape=(train_X.shape[1], train_X.shape[2])))
# 	model.add(Dense(n_series)) # output of size n_series without activation function
# 	model.compile(loss='mean_squared_error', optimizer='adam')
	
# 	# fit network
# 	history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# 	# RMSE
# 	y, y_hat, rmse = calculate_rmse(n_series, n_features, n_lags, test_X, test_y, scaler, model)

# 	last = predict_last(n_series, n_features, n_lags, last_values, scaler, model, 1)

# 	return history, rmse, y, y_hat, last

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
	#print(train_X.shape)
	# insert lag index to data
	train_X = np.insert(train_X, train_X.shape[-1], np.arange(train_X.shape[1])-n_lags, axis=2)
	test_X = np.insert(test_X, test_X.shape[-1], np.arange(test_X.shape[1])-n_lags, axis=2)
	n_features += 1
	# print(train_X)
	#print(train_X[:, :, -3:])
	#print(train_X.shape)
	#raise Exception('debugging')

	trdm = Data_manager(train_X, train_y)
	tedm = Data_manager(test_X, test_y)

	if(len(train_X) % batch_size == 1 or len(test_X) % batch_size == 1):
		batch_size += 1

	trdl = DataLoader(trdm, batch_size=batch_size)
	tedl = DataLoader(tedm, batch_size=len(train_X))

	model = Model(n_hidden, train_X.shape[2], 1, n_series, n_lags)
	opt = optim.Adam(model.parameters(), lr)
	loss_fnc = nn.MSELoss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for epoch in range(n_epochs):
		train_loss = 0
		for X, y in trdl:
			X.to(device)
			y.to(device)

			# opt.zero_grad()
			# pred = model(X)
			# loss = loss_fnc(pred, y)
			# loss.backward()
			# opt.step()
			# train_loss += loss.data.item()

			out = []
			total_loss = 0
			for i in range(n_series):
				pred = model(X)
				loss = loss_fnc(pred.view(-1), y[:, i])
				opt.zero_grad()
				loss.backward()
				opt.step()
				out.append(pred.detach().numpy().ravel())

				# agregar prediccion a la entrada
				X = cambiar_X(X, pred, device)
				#print(X[-1, -2:, :])
			#raise Exception('aca')

			out = np.array(out).T
			out = torch.Tensor(out, device=device)
			loss = loss_fnc(out, y)
			
			train_loss += loss.data.item()			

			# pred, loss = predict_recursively(X, y, model, loss_fnc, opt, n_series, device, True)
			# train_loss += loss

		test_loss = 0
		for X, y in tedl:
			with torch.no_grad():
				X.to(device)
				y.to(device)

				# pred = model(X)
				# loss = loss_fnc(pred, y)
				# test_loss += loss.data.item()

				pred, loss = predict_recursively(X, y, model, loss_fnc, opt, n_series, device, False)
				test_loss += loss

		print('epoch: ', epoch)
		print('train loss: ', train_loss)
		print('test loss: ', test_loss, end='\n\n')

	# last = model(torch.Tensor(np.expand_dims(last_values, axis=0), device=device))
	last, _ = predict_recursively(torch.Tensor(np.expand_dims(np.insert(last_values, last_values.shape[1], 0, axis=1), axis=0), device=device), torch.zeros(1, n_series).to(device), model, loss_fnc, opt, n_series, device, False)

	return test_loss, y.detach().numpy(), pred.detach().numpy(), last.detach().numpy()

def predict_recursively(X, y, model, loss_fnc, opt, n_out, device, train):
	out = []
	total_loss = 0
	for i in range(n_out):
		pred = model(X)
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
	def __init__(self, n_hidden, n_in, n_out, n_series, n_lags):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
		self.n_hidden, self.n_in, self.n_out, self.n_series = n_hidden, n_in, n_out, n_series
		# self.activation = nn.ReLU().to(self.device)
		# self.rnn = nn.LSTM(self.n_in, self.n_hidden, num_layers=self.n_series, batch_first=True).to(self.device)
		self.rnn = nn.LSTM(self.n_in, n_out, num_layers=self.n_series, batch_first=True).to(self.device)
		# self.reduce = nn.Linear(self.n_hidden, n_out).to(self.device)
		# self.out = nn.Linear(n_lags, self.n_series).to(self.device)

	def forward(self, seq):
		# seq = self.activation(seq)
		rnn_out, self.h = self.rnn(seq)
		out = rnn_out[:, -1, :]
		# outp = self.reduce(rnn_out)
		#outp = outp.view(outp.size(0), outp.size(1))
		#out = self.out(outp)
		# out = self.activation(outp)
		return out

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