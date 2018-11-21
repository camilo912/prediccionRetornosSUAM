import pandas as pd
import numpy as np

df = pd.read_csv('data/data_15_11_2018.csv', header=0, index_col=0)

print(df.index.values)
print(df.loc['2017-12-31','JNCAPMOM_Index#@#PX LAST'])