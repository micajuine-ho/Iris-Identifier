# Micajuine Ho
#
# Simple Machine Learning exercise using Scikit-learn and my own basic 
# implmentation of K-Nearest Neighbors on Ronald Fisher's 1936 Iris 
# flower data set. I used supervised learning, specifically KNN, to 
# classify Iris flowers by their species based upon the multivariate 
# data set. The features in the data set are Sepal length, Sepal width, 
# Petal length, and Petal width. The possible species of Iris flowers are
# Setosa, Versicolor, and Virginica.
#
# The iris dataset can be found here: https://en.wikipedia.org/wiki/Iris_flower_data_set

from scipy.spatial import distance 
def dist(a,b):
    return distance.euclidean(a,b)

class My_KNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            predictions.append(self.closest(row))
        return predictions

    def closest(self, row):
        best_index = 0
        best_dist = dist(row, self.x_train[0]) 
        for i in range (1, len(self.x_train)):
            curr_dist = dist (row, self.x_train[i])
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_index = i
        return self.y_train[best_index]

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as nump

# Load data
iris = load_iris()

# Treat response and explanatory variables as f(x) = y
x = iris.data
y = iris.target

# Splits the 2 arrays (x, y) into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# Initialize classifier
classifier = My_KNN() 

# Fit the training data
classifier.fit(x_train, y_train)

# Make predicition using classifier and test data
predicition = classifier.predict(x_test)

# Validation 
print (accuracy_score(predicition, y_test))
