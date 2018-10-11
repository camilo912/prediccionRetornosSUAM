import pandas as pd
import numpy as np

from keras import backend as K
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

	X_train, y_train = data[:n_train, :], data[1:n_train+1]
	X_val, y_val = data[n_train:n_train+n_val, :], data[n_train+1:n_train+n_val+1]
	X_test, y_test = data[n_train+n_val:-1, :], data[n_train+n_val+1:]

	return X_train, X_val, X_test, y_train, y_val, y_test

def weighted_mse(yTrue,yPred):

    ones = K.ones_like(yTrue[0,:]) # a simple vector with ones shaped as (10,)
    idx = K.cumsum(ones) # similar to a 'range(1,11)'

    return K.mean((1/idx)*K.square(yTrue-yPred))


df = pd.read_csv('data/forecast-competition-complete.csv', header=0, index_col=0)

# Parameters
lr = 0.001
lr_decay = 0.0
n_epochs = 100
batch_size = 80

data, scaler = normalize_data(df.values)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

model = Sequential()

model.add(LSTM(X_train.shape[1], input_shape=( None, X_train.shape[1]), return_sequences=True))
model.add(LSTM(X_train.shape[1], return_sequences=True))

opt = Adam(lr=lr, decay=lr_decay)
model.compile(loss=weighted_mse, optimizer=opt)

model.fit(np.expand_dims(X_train, axis=0), np.expand_dims(y_train, axis=0), epochs=n_epochs, batch_size=batch_size, verbose=0, shuffle=False)

new_model = Sequential()
new_model.add(LSTM(X_train.shape[1], batch_input_shape=(1, None, X_train.shape[1]), return_sequences=True, stateful=True))
new_model.add(LSTM(X_train.shape[1], return_sequences=False, stateful=True))

# Transfer learning
new_model.set_weights(model.get_weights())

# Validation
preds_val = []
obs_val = []
for i in range(int(len(df)*0.6), int(len(df)* 0.8), 10):
	pred1 = new_model.predict(np.expand_dims(data[:i], axis=0))
	pred2 = new_model.predict(np.expand_dims(pred1, axis=0))
	pred3 = new_model.predict(np.expand_dims(pred2, axis=0))
	pred4 = new_model.predict(np.expand_dims(pred3, axis=0))
	pred5 = new_model.predict(np.expand_dims(pred4, axis=0))
	pred6 = new_model.predict(np.expand_dims(pred5, axis=0))
	pred7 = new_model.predict(np.expand_dims(pred6, axis=0))
	pred8 = new_model.predict(np.expand_dims(pred7, axis=0))
	pred9 = new_model.predict(np.expand_dims(pred8, axis=0))
	pred10 = new_model.predict(np.expand_dims(pred9, axis=0))

	preds_val.append([pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0], pred5[0][0], pred6[0][0], pred7[0][0], pred8[0][0], pred9[0][0], pred10[0][0]])
	obs_val.extend(data[i:i+10, 0])

# Testing
preds_test = []
obs_test = []
for i in range(int(len(df)*0.8), int(len(df)), 10):
	pred1 = new_model.predict(np.expand_dims(data[:i], axis=0))
	pred2 = new_model.predict(np.expand_dims(pred1, axis=0))
	pred3 = new_model.predict(np.expand_dims(pred2, axis=0))
	pred4 = new_model.predict(np.expand_dims(pred3, axis=0))
	pred5 = new_model.predict(np.expand_dims(pred4, axis=0))
	pred6 = new_model.predict(np.expand_dims(pred5, axis=0))
	pred7 = new_model.predict(np.expand_dims(pred6, axis=0))
	pred8 = new_model.predict(np.expand_dims(pred7, axis=0))
	pred9 = new_model.predict(np.expand_dims(pred8, axis=0))
	pred10 = new_model.predict(np.expand_dims(pred9, axis=0))

	preds_test.append([pred1[0][0], pred2[0][0], pred3[0][0], pred4[0][0], pred5[0][0], pred6[0][0], pred7[0][0], pred8[0][0], pred9[0][0], pred10[0][0]])
	obs_test.extend(data[i:i+10, 0])

# Plots

# validation
plt.title('validation plot')
plt.plot(obs_val, label='observations')
for i in range(len(preds_val)):
	pad = [None for j in range(i*10)]
	plt.plot(pad + list(preds_val[i]), label='forecasts')

plt.legend()
plt.show()

# test
plt.title('Test plot')
plt.plot(obs_test, label='observations')
for i in range(len(preds_test)):
	pad = [None for j in range(i*10)]
	plt.plot(pad + list(preds_test[i]), label='forecasts')

plt.legend()
plt.show()



###############################################################################################################################################

# # ***** For 1 step prediction *****

# preds_val = []
# obs_val = []
# for i in range(int(len(df)* 0.6), int(len(df)* 0.8)):
# 	pred = new_model.predict(np.expand_dims(data[:i, :], axis=0))
# 	preds_val.append(pred[0, 0])
# 	obs_val.append(data[i, 0])

# preds_test = []
# obs_test = []
# for i in range(int(len(df)* 0.8), int(len(df))):
# 	pred = new_model.predict(np.expand_dims(data[:i, :], axis=0))
# 	preds_test.append(pred[0, 0])
# 	obs_test.append(data[i, 0])

# plt.title('validation plot')
# plt.plot(obs_val, label='observations')
# plt.plot(preds_val, label='forecasts')
# plt.legend()
# plt.show()


# plt.title('test plot')
# plt.plot(obs_test, label='observations')
# plt.plot(preds_test, label='forecasts')
# plt.legend()
# plt.show()
