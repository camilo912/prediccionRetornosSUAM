import pandas as pd
import numpy as np

df = pd.read_csv('data/variables_16_11_2018.csv', header=0, index_col=0)
df_regional = pd.read_csv('data/returns_quarterly.csv', header=0, index_col=0)

cols = list(df.columns)

df['AMERICAS'] = df_regional['AMERICAS']


df = df[['AMERICAS'] + cols]

df.to_csv('data/returns_complete_americas.csv')
