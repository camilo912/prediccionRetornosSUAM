import pandas as pd
import numpy as np
import utils
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

## En esta calse se debe hacer un modelo de random forest con el dataset seleccionado y mira por que casi no fluctua cuando se hace en el main.

df = pd.read_csv('data/forecast-competition-complete.csv', header=0, index_col=0)

values, scaler = utils.normalize_data(df.values[:400])

verbose = 0
n_features = values.shape[1]
# # for random forest
# n_lags = 4
# n_series = 1
# n_estimators = 762
# max_features = 18
# min_samples = 3

# for lstm
batch_size = 52
lr = 0.001 # 0.4799370248396754
n_epochs = 33
n_hidden = 159
n_lags = 28
n_series = 1

train_X, val_X, test_X, train_y, val_y, test_y, last_values = utils.transform_values(values, n_lags, n_series, 1)

############# random forest ##############33
# model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, min_samples_leaf=min_samples, n_jobs=-1, verbose=verbose)
# model.fit(train_X, train_y)

############## lstm ###################
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(n_lags, n_features)))
model.add(Dense(n_series))
opt = Adam(lr=lr)
model.compile(loss='mse', optimizer=opt)

model.fit(train_X, train_y, epochs=n_epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=verbose)

y_hat_val = model.predict(val_X)
y_hat = model.predict(test_X)

y_hat_val = utils.inverse_transform(y_hat_val, scaler, n_features)
val_y = utils.inverse_transform(val_y, scaler, n_features)

plt.plot(val_y, label='observations')
plt.plot(y_hat_val, label='predictions')
plt.suptitle('validation plot')
plt.legend()
plt.show()

# y_hat = utils.inverse_transform(y_hat, scaler, n_features)
# test_y = utils.inverse_transform(test_y, scaler, n_features)

# plt.plot(test_y, label='observations')
# plt.plot(y_hat, label='predictions')
# plt.suptitle('test plot')
# plt.legend()
# plt.show()

print('rmse: ', mean_squared_error(val_y, y_hat_val)**(1/2))
#print('rmse: ', mean_squared_error(test_y, y_hat)**(1/2))

