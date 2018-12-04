import pandas as pd
import numpy as np

df = pd.read_csv('data/data_16_11_2018.csv', header=0, index_col=0)

df[list(df.columns)[0]] = df[list(df.columns)[0]].pct_change()

df.dropna(inplace=True)
df.index = np.arange(len(df))

df.to_csv('data/data_16_11_2018_differentiated.csv')