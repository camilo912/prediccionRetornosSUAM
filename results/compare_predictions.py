import pandas as pd
from sklearn.metrics import mean_squared_error
import math
from os import listdir

df = pd.read_csv('forecast-competition-complete.csv')
df = df.loc[400:500, 'TARGET']

rmses = []
names = [f for f in listdir('.') if f[:10] == 'resultados']
print(names)
# names = ['resultados.csv', 'resultadosSarimax.csv', 'resultadosArima.csv', 'resultadosAdaBoost.csv', 'resultadosAdaBoost2.csv', 'resultadosLSTM.csv', 'resultadosRandomForest.csv', 'resultadosRandomForest2.csv', 'resultadosSVMlinear.csv', 'resultadosSVMpoly1.csv', 'resultadosSVMpoly12.csv', 'resultadosSVMpoly2.csv', 'resultadosSVMpoly3.csv', 'resultadosSVMpoly4.csv', 'resultadosSVMpoly5.csv', 'resultadosSVMrbf.csv', 'resultadosSVMsigmoid.csv']
# names = ['resultadosAdaBoost.csv', 'resultadosRandomForest.csv', 'resultadosSVMlinear.csv', 'resultadosSVMpoly1.csv', 'resultadosSVMpoly2.csv', 'resultadosSVMpoly3.csv', 'resultadosSVMpoly4.csv', 'resultadosSVMpoly5.csv', 'resultadosSVMrbf.csv', 'resultadosSVMsigmoid.csv']
maxi_c = ''
maxi = 1000000000000000
table = {}
print('\n\nalgoritmo \t\t\t|\t rmse\n' + '-'*32 + '+' + '-'*28)
for n in names:
	dfo = pd.read_csv(n, header=-1)
	if(dfo.values.shape[0] > df.values.shape[0]):
		print(dfo)
	rmse = math.sqrt(mean_squared_error(df.values, dfo.values))
	table[n[10:-4]] = rmse
	# if(len(n[10:-4]) < 6):
	# 	print(n[10:-4] + ': \t\t\t' + str(rmse))
	# else:
	# 	print(n[10:-4] + ': \t\t' + str(rmse))
	rmses.append(rmse)
	if(rmse < maxi):
		maxi = rmse
		maxi_c = n

for key, value in sorted(table.items(), key= lambda x: x[1]):
	if(len(key) < 6):
		print("%s: \t\t\t\t|\t %s" % (key, value))
	elif(len(key) < 17):
		print("%s: \t\t\t|\t %s" % (key, value))
	elif(len(key) < 24):
		print("%s: \t\t|\t %s" % (key, value))
	else:
		print("%s: \t|\t %s" % (key, value))

print('\n\nel mejor fue: %s con %f' % (maxi_c[10:-4], maxi))
# print(rmses)
