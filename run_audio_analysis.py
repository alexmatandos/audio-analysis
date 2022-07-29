import pandas
import kfold_template
from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("dataset.csv")
dataset = dataset.drop(['filename'], axis = 1)
dataset = dataset.sample(frac = 1).reset_index(drop = True)

target = dataset.iloc[:, -1].values
data = dataset.iloc[:, : -1].values

print(target)
print(data)

target, key = pandas.factorize(target)
print(key)

machine = RandomForestClassifier(criterion = "gini", max_depth = 100, n_estimators = 300, bootstrap = True, max_features = "auto")
results = kfold_template.run_kfold(data, target, 4, machine, 1, 1, 1)

print(results[1])
for result in results[2]:
	print(result)