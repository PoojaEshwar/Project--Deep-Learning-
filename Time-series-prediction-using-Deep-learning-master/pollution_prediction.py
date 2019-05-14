import pandas
from sklearn import preprocessing
import numpy as np
from keras.models import load_model
from pandas import read_csv
from matplotlib import pyplot

dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pandas.DataFrame(data)
	cols, names = list(), list()

	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	agg = pandas.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

x=11495
y=11505
encoder = preprocessing.LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values = values.astype('float32')
print(values[x:y,0])
s=values[x:y].std(axis=0)
m=values[x:y].mean(axis=0)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[x:y, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model=load_model("airpollution_model.h5")
test=test_X[0:9]
print(test[:,0][:,1])
yhat = model.predict(test)
print("------------------")
print(yhat[:,0][7])
print("==================")
np.savetxt('ip.txt',test[:,0][:,0])
np.savetxt('test.txt',yhat[:,0], delimiter=',')
pyplot.plot(test[:,0][:8,0],label="input")
pyplot.plot(yhat[:,0],label="prediction")
pyplot.axvline(x=7)
pyplot.show()
res=yhat[:,0][7]
res=res*s+m
export=res[0]
print(res[0])

