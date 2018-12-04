import pandas as pd
import numpy as np

# df = pd.read_csv('data/data_selected.csv', header=0, index_col=0)

# f = open('selected_variables.txt', 'a')

# f.write(','.join(list(df.columns)).replace('#@#PX_LAST', '')+ '\n')

# f.close()

f = open('selected_variables.txt', 'r')
lines = f.readlines()
lines = [line.strip().split(',') for line in lines]

common = list(set(lines[0]).intersection(*lines))

print(common)
print(len(common))

f.close()