from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as nump

iris = load_iris()
# print(iris.data)
# print(iris.target)

#Validation data
validate_index = [0, 50, 100]
validate_target = iris.target[validate_index]
validate_data = iris.data[validate_index]

#training data
training_data = nump.delete(iris.data, validate_index, axis=0)
training_target = nump.delete(iris.target, validate_index)

#Classifier 
tree_classifier = tree.DecisionTreeClassifier() 
tree_classifier.fit(training_data, training_target)

#Classifier prediction
prediction = tree_classifier.predict(validate_data)

#Verification
print (accuracy_score(validate_target, prediction))
