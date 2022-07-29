import numpy
import pandas

from sklearn.preprocessing import StandardScaler

import keras
from keras import layers
from keras.models import Sequential

from sklearn.model_selection import KFold
from sklearn import metrics

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.drop(['filename'], axis = 1)
dataset = dataset.sample(frac = 1).reset_index(drop = True)

target = dataset.iloc[:, -1].values

data = dataset.iloc[:, : -1].values
scaler = StandardScaler()
data = scaler.fit_transform(numpy.array(data, dtype = float))

print(target)
print(data)

target, key = pandas.factorize(target)
print(key)

kfold_object = KFold(n_splits = splits_number)
kfold_object.get_n_splits(data)

result_accuracy = []
result_confusion = []

for training_index, test_index in kfold_object.split(data):
	data_training = data[training_index]
	target_training = target[training_index]
	data_test = data[test_index]
	target_test = target[test_index]

	machine = Sequential()
	machine.add(layers.Dense(256, activation = "relu", input_shape = ))