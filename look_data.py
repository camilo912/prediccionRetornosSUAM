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

# two graphs per window
for i in range(0, len(df.columns), 2):
	plt.subplot(2, 1, 1)
	plt.plot(df[list(df.columns)[i]])
	plt.subplot(2, 1, 2)
	plt.plot(df[list(df.columns)[i+1]])
	plt.show()
