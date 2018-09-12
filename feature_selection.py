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
	# Stepwise forward selection
	# params

	n_news -= 1
	features = set(dataFrame.columns)
	features.remove(list(dataFrame.columns)[0])
	missing = features.copy()
	inside = [list(dataFrame.columns)[0]]

	from sklearn.tree import DecisionTreeRegressor
	while(n_news):
		fts = list(inside)
		best = ''
		best_importance = 0
		for ft in missing:
			fts = fts + [ft]
			scaled, scaler = normalize_data(dataFrame[fts].values)
			x, y = series_to_supervised(scaled)
			model = DecisionTreeRegressor()
			model.fit(x, y)
			importances = model.feature_importances_
			if(importances[-1] > best_importance):
				best = fts[-1]
				best_importance = importances[-1]
		
		inside.append(best)
		missing.remove(best)

		n_news -= 1

	df = dataFrame[inside]
	df.to_csv('forecast-competition-complete_selected.csv')

def select_features_ga(dataFrame):
	n_generations = 12
	n_chars = dataFrame.shape[1]
	n_villagers = max(1, int(n_chars / 2))
	villagers = np.random.randint(2, size=(n_villagers, n_chars))
	villagers = np.array([[bool(villagers[i,j]) for i in range(villagers.shape[0])] for j in range(villagers.shape[1])]).T
	n_best_parents = int(n_villagers / 2)

	columns = np.array(dataFrame.columns)
	for generation in range(n_generations):
		print('generation: %d' % (generation + 1))
		losses = []
		for villager in villagers:
			# asure that target variable is in the solution
			villager[0] = True

			cols = columns[villager]
			df = dataFrame[cols]

		

			import predictor
			from sklearn.tree import DecisionTreeRegressor
			model = DecisionTreeRegressor()

			scaled, scaler = predictor.normalize_data(df.values)
			values, n_lags, n_series = scaled, 4, 1
			train_X, test_X, train_y, test_y, last_values = predictor.transform_values(values, n_lags, n_series, 0)
			model.fit(train_X, train_y)
			pred = model.predict(test_X)

			from sklearn.metrics import mean_squared_error
			loss = mean_squared_error(pred, test_y)
			losses.append(loss)

			# # space for dimensional reduction
			# import dimensionality_reduction
			# scaled = dimensionality_reduction.reduce_dimensionality(scaled, 10)

			# n_features = scaled.shape[1]

			# batch_size, lr, n_epochs, n_hidden, n_lags, weight_decay = 75, 0.13575862699150665, 153, 167, 30, 0.00005

			# data = lstm.series_to_supervised(scaled, n_lags, 1, ys=ys)
			# train_X, test_X, train_y, test_y = lstm.split_data(data, n_features, n_lags)

			# loss_function = lstm.get_loss_function()
			# model = lstm.Model(train_X.shape[1], n_hidden)
			# optimizer = lstm.get_optimizer(model, lr, weight_decay)

			# _, loss = lstm.fit(train_X, test_X, train_y, test_y, n_hidden, n_epochs, loss_function, optimizer, model, batch_size, weight_decay, 0)
			# losses.append(loss)

		losses = np.array(losses)
		# print(losses)
		# print('best loss: %f' % np.min(losses))
		# Select best parents
		parents = []
		for n in range(n_best_parents):
			idx = np.where(losses == np.min(losses))[0][0]
			parents.append(villagers[idx])
			losses[idx] = np.NINF

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

		villagers[:n_best_parents] = parents
		villagers[n_best_parents:] = cross_over

	cols = columns[villagers[np.where(losses == np.min(losses))[0][0]]]
	df = dataFrame[cols]
	df.to_csv('forecast-competition-complete_selected.csv')




if __name__ == '__main__':
	df = pd.read_csv('series_energ_v2.csv', index_col=0)
	select_features(df, 30)