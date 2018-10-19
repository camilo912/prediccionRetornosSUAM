import numpy as np
import pandas as pd

def normalize_data(data, scale=(0,1)):
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=scale)
	scaled = scaler.fit_transform(data)
	return scaled, scaler

def split_data(series):
	wall = int(series.shape[0] * 0.8) 
	train_X = series[:wall, :]
	test_X = series[wall:-1, :]
	train_y = series[1:wall+1, 0]
	test_y = series[wall+1:, 0]
	return train_X, test_X, train_y, test_y

def series_to_supervised(series):
	x = series[:-1, :]
	y = series[1:, 0]
	return x, y

def select_features_stepwise_forward(dataFrame, n_news=25):
	n_features = dataFrame.shape[1]

	# params
	n_news -= 1
	features = set(dataFrame.columns)
	features.remove(list(dataFrame.columns)[0])
	missing = features.copy()
	inside = [list(dataFrame.columns)[0]]
	from sklearn.ensemble import RandomForestRegressor

	while(n_news):
		fts = list(inside)
		best = ''
		best_importance = 0
		for ft in missing:
			fts = fts + [ft]
			scaled, scaler = normalize_data(dataFrame[fts].values)
			x, y = series_to_supervised(scaled)
			model = RandomForestRegressor()
			model.fit(x, y)
			importances = model.feature_importances_
			if(importances[-1] > best_importance):
				best = fts[-1]
				best_importance = importances[-1]
		
		inside.append(best)
		missing.remove(best)

		n_news -= 1

	df = dataFrame[inside]
	df.to_csv('data/forecast-competition-complete_selected.csv')

def select_features_ga(dataFrame, max_vars, original_cols):
	import predictor
	import random
	import time
	from matplotlib import pyplot as plt
	from sklearn.metrics import mean_squared_error
	#from sklearn.ensemble import RandomForestRegressor
	#from sklearn.linear_model import LinearRegression
	from sklearn.svm import SVR
	n_generations = 250
	n_chars = dataFrame.shape[1]
	n_villagers = max(1, int(n_chars / 2))
	villagers = np.random.randint(2, size=(n_villagers, n_chars))
	villagers = np.array([[bool(villagers[i,j]) for i in range(villagers.shape[0])] for j in range(villagers.shape[1])]).T
	n_best_parents = int(n_villagers / 2)
	historic_losses = []
	print_gen = int(n_generations / 10)

	columns = np.array(dataFrame.columns)
	for generation in range(n_generations):
		# start_time = time.time()
		if((generation + 1) % print_gen == 0): print('generation: %d of %d' % (generation + 1, n_generations))
		losses = []
		for villager in villagers:
			# asure that target variable is in the solution
			villager[0] = True

			cols = columns[villager]
			df = dataFrame[cols]

			#model = LinearRegression(n_jobs=-1)
			#model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
			model = SVR(gamma='scale')
			#model = SVR(kernel='linear')

			scaled, scaler = predictor.normalize_data(df.values)
			values, n_lags, n_series = scaled, 4, 1
			train_X, val_X, test_X, train_y, val_y, test_y, last_values = predictor.transform_values(values, n_lags, n_series, 0)
			model.fit(train_X, train_y.ravel())
			pred = model.predict(val_X)

			loss = mean_squared_error(pred, val_y)
			losses.append(loss)

		losses = np.array(losses)
		temp_losses = losses.copy()
		historic_losses.append(np.min(losses))
		
		# Select best parents
		parents = []
		for n in range(n_best_parents):
			idx = np.where(temp_losses == np.min(temp_losses))[0][0]
			parents.append(villagers[idx])
			temp_losses[idx] = np.Inf

		# Cross over
		cross_over = []
		one_point = int(n_chars / 2)
		for n in range(n_villagers - n_best_parents):
			tmp = np.zeros(n_chars)
			tmp[:one_point] = parents[n % len(parents)][:one_point]
			tmp[one_point:] = parents[(n + 1) % len(parents)][one_point:]
			cross_over.append(tmp)

		# Mutation
		for i in range(len(cross_over)):
			for j in range(n_chars):
				if(np.random.rand() < 1.0/n_chars):
					cross_over[i][j] = not(cross_over[i][j])

		# Max vars trim
		for i in range(len(cross_over)):
			suma = np.sum(cross_over[i])
			if(suma > max_vars):
				# how many to drop
				leftover = int(suma - max_vars)
				# take the positives
				positives = [i for i, x in enumerate(cross_over[i]) if x]
				# select the ones to drop
				random.shuffle(positives)
				positives = positives[:leftover]
				# drop those ones
				cross_over[i][positives] = False


		villagers[:n_best_parents] = parents
		villagers[n_best_parents:] = cross_over

	cols = columns[villagers[np.where(losses == np.min(losses))[0][0]]]
	df = dataFrame[cols]
	df.columns = np.array(original_cols)[cols]
	df.to_csv('data/forecast-competition-complete_selected.csv')

def select_features_sa(dataFrame, max_vars, original_cols):
	from simanneal import Annealer

	class Sas(Annealer):
		def __init__(self, state, df):
			self.df = df.copy()
			super(Sas, self).__init__(state)

		def move(self):
			idx = np.random.randint(0, len(self.state))
			self.state[idx] = not(self.state[idx])

		def energy(self):
			from sklearn.metrics import mean_squared_error
			#from sklearn.linear_model import LinearRegression
			#model = LinearRegression()
			#from sklearn.ensemble import RandomForestRegressor
			#model = RandomForestRegressor()
			from sklearn.svm import SVR
			#model = SVR(kernel='poly', degree=1)
			model = SVR(gamma='scale')
			#from sklearn.ensemble import AdaBoostRegressor
			#model = AdaBoostRegressor()
			
			cols = list(self.df.columns[self.state])
			df = self.df[cols]
			data = df.values
			wall = int(len(data) * 0.8)
			X = data[:-1,:]
			X_train = X[:wall]
			X_test = X[wall:]
			y = data[1:, 0]
			y_train = y[:wall]
			y_test = y[wall:]

			model.fit(X_train, y_train)
			pred = model.predict(X_test)

			mse = mean_squared_error(pred, y_test)
			return mse

	n_chars = dataFrame.shape[1]
	initial = np.random.randint(2, size=n_chars)
	initial = np.array([bool(x) for x in initial])
	ga = Sas(initial, dataFrame)
	ga.steps = 10000
	ga.copy_strategy = "slice"
	result, mse = ga.anneal()
	cols = np.array(dataFrame.columns)
	cols = cols[result]
	df = dataFrame[cols]
	df.columns = np.array(original_cols)[cols]
	df.to_csv('data/forecast-competition-complete_selected.csv')


if __name__ == '__main__':
	df = pd.read_csv('series_energ_v2.csv', index_col=0)
	select_features(df, 30)