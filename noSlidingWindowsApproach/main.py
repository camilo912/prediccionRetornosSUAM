import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

def normalize_data(data, scale=(-1, 1)):
	scaler = MinMaxScaler(feature_range=scale)
	scaled = scaler.fit_transform(data)
	return scaled, scaler

def split_data(data):
	n_train = int(len(data)*0.6)
	n_val = int(len(data)*0.2)
	ys = data[1:, 0]

	# X_train, y_train = data[:n_train, :], ys[:n_train]
	# X_val, y_val = data[n_train:n_train+n_val, :], ys[n_train:n_train+n_val]
	# X_test, y_test = data[n_train+n_val:-1, :], ys[n_train+n_val:]

	X_train, y_train = data[:n_train, :], data[1:n_train+1]
	X_val, y_val = data[n_train:n_train+n_val, :], data[n_train+1:n_train+n_val+1]
	X_test, y_test = data[n_train+n_val:-1, :], data[n_train+n_val+1:]

	return X_train, X_val, X_test, y_train, y_val, y_test


df = pd.read_csv('data/forecast-competition-complete.csv', header=0, index_col=0)

# Parameters
lr = 0.001
lr_decay = 0.0
n_epochs = 100
batch_size = 80

data, scaler = normalize_data(df.values)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

model = Sequential()

model.add(LSTM(X_train.shape[1], input_shape=(None, X_train.shape[1]), return_sequences=True))
model.add(LSTM(X_train.shape[1], return_sequences=True))

opt = Adam(lr=lr, decay=lr_decay)
model.compile(loss='mse', optimizer=opt)

model.fit(np.expand_dims(X_train, axis=0), np.expand_dims(y_train, axis=0), epochs=n_epochs, batch_size=batch_size, verbose=0, shuffle=False)

new_model = Sequential()
new_model.add(LSTM(X_train.shape[1], input_shape=(None, X_train.shape[1]), return_sequences=True))
new_model.add(LSTM(X_train.shape[1], return_sequences=False))

# Transfer learning
new_model.set_weights(model.get_weights())

# opt2 = Adam(lr=lr, decay=lr_decay)
# new_model.compile(loss='mse', optimizer=opt2)
#new = new_model.predict(np.expand_dims(np.expand_dims(X_test[0, :], axis=0), axis=0))

preds_val = []
obs_val = []
for i in range(int(len(df)* 0.6), int(len(df)* 0.8)):
	pred = new_model.predict(np.expand_dims(data[:i, :], axis=0))
	preds_val.append(pred[0, 0])
	obs_val.append(data[i, 0])

preds_test = []
obs_test = []
for i in range(int(len(df)* 0.8), int(len(df))):
	pred = new_model.predict(np.expand_dims(data[:i, :], axis=0))
	preds_test.append(pred[0, 0])
	obs_test.append(data[i, 0])

plt.title('validation plot')
plt.plot(obs_val, label='observations')
plt.plot(preds_val, label='forecasts')
plt.legend()
plt.show()


plt.title('test plot')
plt.plot(obs_test, label='observations')
plt.plot(preds_test, label='forecasts')
plt.legend()
plt.show()
