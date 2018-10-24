import pandas as pd
import numpy as np
import predictor
from matplotlib import pyplot as plt

df = pd.read_csv('data/forecast-competition-complete.csv', header=0, index_col=0)

# Parameters
model = 5 # id of model to use
parameters = 0 # Set to True for performing bayes optimization looking for best parameters
select = 0 # set to True for performing feature selection
original = 1 # set to True for training with original data (not feature selected)
time_steps = 10 # number of periods in the future to predict
max_vars = 50 # maximum number of variables for taking in count for variable selection
plots_level = 1 # level of log plots
########## niveles de plot tambien para testing
parameters_file_name = 'parameters/default_lstm_10timesteps.txt'

cols = list(df.columns)
########### hacer argumento para parametros  con nombre de archivo y uno por default ******************** CHECKED
######## hacer argumento para archivo de varaibles seleccioandas igual que apra el de leer parametros *** hecho con el archivo de inico
########### hacer optimizacion bayesiana con el error de validacion y no el de testing ************ CHECKED
pred = [predictor.predictor(df.values[:i, :], cols, model, parameters, select, original, time_steps, max_vars, plots_level, parameters_file_name) for i in range(400, 460, time_steps)]

plt.plot(df.values[400:, 0], label='observations')
if(time_steps == 1):
	plt.plot(pred, label='predictions', marker='*')
elif(time_steps > 1):
	for i in range(len(pred)):
		pad = [None for j in range(i*time_steps)]
		plt.plot(pad + list(pred[i]), label='predictions', marker='*')
plt.legend()
plt.show()