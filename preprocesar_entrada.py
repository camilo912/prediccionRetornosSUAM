import pandas as pd 
import numpy as np

def get_consistent_columns(df, target):
	"""
		Función que a partir de un dataframe y una variable objetivo retorna un nuevo dataframe con variables que tengan valores en las mismas fechas de la variable objetivo, teniendo en cuenta
		la periodicidad.

		Parámetros:
		- df -- DataFrame de pandas, dataframe con los datos de todas las variables
		- target -- String, nombre de la variable objetivo

		Retorna:
		- df_new -- DataFrame de pandas, dataframe con la varaible objetivo y las variables que sean consitentes con las fehcas de la variable objetivo teniendo en cuetna la periodicidad.
	"""
	consistent = [target]
	serie_taget = df[target]
	serie_taget.dropna(inplace=True)
	df_new = pd.DataFrame(serie_taget)

	# para cada columna (cada variable)
	for c in df.columns:
		# que no sea la varaible objetivo
		if(c != target):
			# elimina los valores nulos
			col = df[c].dropna(inplace=False)
			# elimina los valores duplicados, mantiene el último que observe
			col = col[~col.index.duplicated(keep='last')]
			# si en las fechas de la variable objetivo no hay datos nulos entonces esa variable tiene datos consistentes con la variable objetivo, por lo que se agrega al nuevo DF
			if(not col[serie_taget.index].isnull().values.any()):
				df_new[c] = col

	#print(df_new.index.values)
	#print(df_new.isnull().values.any())
	#print(df_new.shape)
	return df_new

def main():
	"""
		Main del archivo, este archivo se encarga de preprocesar los datos que se bajaron de la base de datos y estandarizarlos a datos consistentes según la variable que se quiera predecir.
		Se tiene en cuetna la variable que se desea redecir y el periodo que se desea predecir para obtener las variables que son consistentes.
	"""
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