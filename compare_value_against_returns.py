import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utils

def transform_from_returns_to_values(returns, bases):
	return bases * (1 + returns)

def convert_to_float(string):
	string = string.replace('[', '').replace(']', '')
	return float(string)

df_returns = pd.read_csv('results/salida_1_periodos_SPTR.csv', header=0, index_col=0)
df_values = pd.read_csv('results/salida_1_periodos.csv', header=0, index_col=0)
df_observations = pd.read_csv('data/data_16_11_2018.csv', header=0, index_col=0)
df_observations.index = np.arange(len(df_observations))
bases = df_observations.loc[200:224, list(df_observations.columns)[0]].values
observations_raw = df_observations.loc[200:, list(df_observations.columns)[0]].values
df_observations = df_observations.loc[201:, list(df_observations.columns)[0]]

# change input from string to float
df_returns[list(df_returns.columns)[0]] = df_returns[list(df_returns.columns)[0]].apply(convert_to_float)
df_values[list(df_values.columns)[0]] = df_values[list(df_values.columns)[0]].apply(convert_to_float)

returns_raw = df_returns.values.ravel()
returns = transform_from_returns_to_values(df_returns.values.ravel(), bases)
values = df_values.values.ravel()
observations = df_observations.values

# print(returns.shape)
# print(values.shape)
# print(observations.shape)

plt.subplot(2, 1, 1)
plt.plot(returns, label='returns', color='red')
plt.plot(values, label='values', color='blue')
plt.plot(observations, label='observations', color='green')
plt.legend()
#plt.show()

plt.subplot(2, 2, 3)
dir_accs = [utils.get_returns_values_direction_accuracy(observations_raw, returns_raw), utils.get_direction_accuracy(observations, values)]
plt.bar(['returns', 'values'], dir_accs)
plt.show()