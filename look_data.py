import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('data/forecast-competition-complete.csv', index_col=0, header=0)
# # all graphs in a window
# n_rows = int(len(df.columns)**(1/2)) # sqrt of len of cols
# n_cols = n_rows + (len(df.columns) % n_rows) # en of cols by n_rows more the residuous of len of cols by n_rows

# for i in range(len(df.columns)):
# 	plt.subplot(n_rows, n_cols, i+1)
# 	plt.plot(df[list(df.columns)[i]])

# plt.show()

# Describe and line plot
print(df[list(df.columns)[0]].describe())
plt.plot(df[list(df.columns)[0]])
plt.show()

# histogram and distribution plot
plt.figure(1)
plt.subplot(211)
df[list(df.columns)[0]].hist()
plt.subplot(212)
df[list(df.columns)[0]].plot(kind='kde')
plt.show()

# box and whisker plot
groups = pd.DataFrame()
groups['1-100'] = df.loc[:100, list(df.columns)[0]].values
groups['100-200'] = df.loc[101:200, list(df.columns)[0]].values
groups['200-300'] = df.loc[201:300, list(df.columns)[0]].values
groups['300-400'] = df.loc[301:400, list(df.columns)[0]].values
groups['400-500'] = df.loc[401:, list(df.columns)[0]].values
groups.boxplot()
plt.show()

# check stationarity
from statsmodels.tsa.stattools import adfuller

result = adfuller(df[list(df.columns)[0]])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# auto correlation function and partial auto correlation function
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot

pyplot.figure()
pyplot.subplot(211)
plot_acf(df[list(df.columns)[0]], ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(df[list(df.columns)[0]], ax=pyplot.gca())
pyplot.show()

# # two graphs per window
# for i in range(0, len(df.columns), 2):
# 	plt.subplot(2, 1, 1)
# 	plt.plot(df[list(df.columns)[i]])
# 	plt.subplot(2, 1, 2)
# 	plt.plot(df[list(df.columns)[i+1]])
# 	plt.show()
