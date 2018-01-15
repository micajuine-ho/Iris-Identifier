# Micajuine Ho
#
# Simple Machine Learning exercise using Scikit-learn on Ronald Fisher's 1936
# Iris flower data set. I used supervised learning, specifically a decision 
# tree, to classify Iris flowers by their species based upon the multivariate 
# data set. The features in the data set are Sepal length, Sepal width, 
# Petal length, and Petal width. The possible species of Iris flowers are
# Setosa, Versicolor, and Virginica.
#
# The iris dataset can be found here: https://en.wikipedia.org/wiki/Iris_flower_data_set

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as nump

iris = load_iris()

# Iris data -> .data = explanatory vbs, .target = response vbs.
# print(iris.data)
# print(iris.target)

# Validation data
validate_index = [0, 50, 100]
validate_target = iris.target[validate_index]
validate_data = iris.data[validate_index]

# Training data
training_data = nump.delete(iris.data, validate_index, axis=0)
training_target = nump.delete(iris.target, validate_index)

# Classifier 
tree_classifier = tree.DecisionTreeClassifier() 
tree_classifier.fit(training_data, training_target)

# Classifier prediction
prediction = tree_classifier.predict(validate_data)

# Verification
print (accuracy_score(validate_target, prediction))
