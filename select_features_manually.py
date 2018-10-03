import pandas as pd

df = pd.read_csv('data/forecast-competition-complete.csv', index_col=0, header=0)
print(df.corr())

features = ['TARGET', 'KHFWG', 'KQDOC', 'HSIOM', 'BRYWC', 'XUKCD', 'EFBBA', 'XRSXD', 'HBOKK', 'ZARXQ', 'FPSXN', 'ODZWJ']

df = df[features]

df.to_csv('data/forecast-competition-complete_selected_manually.csv')