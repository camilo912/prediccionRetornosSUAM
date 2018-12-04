import pandas as pd 
import numpy as np

def get_consistent_columns(df, target):
	consistent = [target]
	serie_taget = df[target]
	serie_taget.dropna(inplace=True)
	df_new = pd.DataFrame(serie_taget)

	for c in df.columns:
		if(c != target):
			col = df[c].dropna(inplace=False)
			col = col[~col.index.duplicated(keep='last')]
			if(not col[serie_taget.index].isnull().values.any()):
				df_new[c] = col

	#print(df_new.index.values)
	#print(df_new.isnull().values.any())
	#print(df_new.shape)
	return df_new

def main():
	df = pd.read_csv('data/variables_16_11_2018.csv', header=0, index_col=0)
	# df = pd.read_csv('data/returns_complete_americas.csv', header=0, index_col=0)
	# target = 'IBOXIG_Index#@#PX_LAST' # 1
	# target='IDCOT3TR_Index#@#PX_LAST' # 2
	# target = "IBOXHY_Index#@#PX_LAST"
	# target = "GBIEMCOR_Index#@#PX_LAST"
	# target = "JPEICORE_Index#@#PX_LAST"
	target = 'SPTR_Index#@#PX_LAST'
	period = 'M' # 'Q'

	df.index = pd.DatetimeIndex(df.index).to_period(period)
	df = get_consistent_columns(df, target)
	print(df.shape)
	df.to_csv('data/data_16_11_2018.csv')
	# df.to_csv('data/data_returns.csv')


if __name__ == '__main__':
	main()