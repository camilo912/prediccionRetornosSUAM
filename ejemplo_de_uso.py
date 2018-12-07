import pandas as pd
import numpy as np
import series_predictor
from matplotlib import pyplot as plt

# 12 Parámetros principales
model = 0 # id del modelo a usarse, en este caso 0 para usar redes neuronales
parameters = 0 # controla si se hace optimización de parámetros o no, en este caso no.
select = 0 # controla si se hace selección de variables o no, en este jemplono tiene sentido ya que los datos solo tienen una variable
original = 1 # controla si se utilizan las variables originales o las que se seleccionaron mediante el proceso de selecciòn de variables (las que quedan guardadas en data/data_selected.csv),
			 # en este caso usaremos las varaibles originales por lo que lo ponemos en 1 o True
time_steps = 1 # numero de periodos en el futuro a predecir
max_vars = 200 # Número máximo de varaibles que pueden ser seleccionadas por el metodo de seleción de variables en este caso no importa por que no se realizará este proceso.
verbosity = 0 # nivel de verbosidad de la ejecución.
parameters_file_name = 'parameters/ejemplo_de_uso_params.pars' # archivo de parámetros que se van a leer, se puede poner en None si quiere que se utilicen los parámetros por default que están en el constructor de la clase Predictor.
MAX_EVALS = 50 # número de evaluciones de la optimización bayesiana para encontrar los mejoresparámetros, en este caso no se va a hacer optimización bayesiana por lo que este parámetro no importa
saved_model = False # controla si se quiere entrenar un modelo o leer uno ya entrenado de la carpeta models, en este caso vamos a entrenar uno desde cero por eso ponemos False
model_file_name = 'models/ejemplo_de_uso.h5' # nombre del archivo donde se quiere cargar o guardar el modelo que se va a entrenar o a leer. h5 es la extensión que usa keras para guardar modelos.
returns = False # indica si se está trabajando con retornos, en este caso no

# una vez establecidos los 12 parámetros principalesprocedemos con la ejecución del programa

# primero necesitamos leer los datos:
input_file_name = 'data/data_SPTR_monthly.csv'
dataframe = pd.read_csv(input_file_name, header=0, index_col=0)

# cambiamos el indice del dataframe de fechas a números consectivos para poder separarlo en train y test
dataframe.index = np.arange(len(dataframe))

# calculamos cuantos ejemplos tiene el set de entrenamiento
wall = int(len(dataframe)*0.8)
# obtenemos los ejemplos de entrenamiento
train = dataframe.loc[:wall, :]
# obtenemos las columnas originales del dataset
cols = dataframe.columns

# instanciamos la clase Predictor, la cual se encarga de entrenar y predecir con los modelos.
predictor = series_predictor.Predictor(dataframe.values, model, original, time_steps, train, cols, parameters, select, max_vars, verbosity, parameters_file_name, MAX_EVALS, saved_model, model_file_name, returns)

# iteramos por todos los ejemplos de prueba excepto el último para poder hacer la prueba
preds = []
obs = []
for i in range(wall, len(dataframe)-1, 1):
	# hacemos una predicción y la guardamos
	preds.append(predictor.predict(dataframe.values[[i]])[0]) # se usa el doble [] en los valores del dataframe para que los valores que devuelve queden en 2D, 
															  # que es el formato necesario para la red neuronal recurrente
	# obtenemos un valor contra el cual comparar la predicción
	obs.append(dataframe.values[i+1][0])

# graficamos las predicciones y las observaciones
plt.plot(obs, label='observaciones')
plt.plot(preds, label='predicciones')
plt.legend()
plt.show()
