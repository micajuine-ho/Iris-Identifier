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
import heapq 
import collections 

def dist(a,b):
    return distance.euclidean(a,b)

class My_KNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k):
        predictions = []
        for row in x_test:
            predictions.append(self.closest(row, k))
        return predictions

    def closest(self, row, k):
        #No built in library for max heaps, so just use negative distance
        max_heap = []
        most_occuring_targets = []

        #Add the first k elements to heap, then for the rest of the elms
        #check if the distance is smaller (larger) than the max (min), 
        #and add it into the max (min) heap.
        for i in range (0, len(self.x_train)):

            #Turn distance to a negative so that we can maintain min heap
            curr_dist = -1 * (dist (row, self.x_train[i]))

            #Add first K elements in 
            if i < k:
                heapq.heappush(max_heap, (curr_dist, i)) 
            
            #Check if min element (max distance) is smaller than the curr dist
            elif curr_dist > max_heap[0][0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (curr_dist, i))

        #Add the most occuring target to an array to find the most occuring one.
        for j in range (0, len(max_heap)):
            most_occuring_targets.append(self.y_train[max_heap[j][1]])

        #Return most occuring target
        return collections.Counter(most_occuring_targets).most_common(1)[0][0]

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
predicition = classifier.predict(x_test, 3)

# print(predicition)
# print(y_test)

# Validation 
print (accuracy_score(predicition, y_test))
