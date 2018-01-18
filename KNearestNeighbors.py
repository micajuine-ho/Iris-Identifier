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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from numpy import argmax

# Load data
iris = load_iris()

# Treat response and explanatory variables as f(x) = y
x = iris.data
y = iris.target

# Initialize classifier
classifier = My_KNN() 

# Store the averages in an array
averages_array = []

# Hyperparameter tuning to choose K using K-Fold cross validation
# Low values of k are prone to overfitting, while high values of k
# are succeptible to high bais. 
for k in range (1,len(x)//2):
    # The 150 data points will be tested in sections of 30
    # Important to shuffle here so that the iris data is split into random groups
    kf = KFold(n_splits=5, shuffle=True, random_state = 12882) 
    average_prediction_score = 0
    for (train_index, test_index) in kf.split(x):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the training data
        classifier.fit(x_train, y_train)

        # Make predicition using classifier and test data
        predicition = classifier.predict(x_test, k)

        # Find accuracy score add to -runing average
        average_prediction_score = average_prediction_score + accuracy_score(predicition, y_test)
    
    #Take the average of the accuracy of all the predictions and append to array 
    averages_array.append(average_prediction_score/5)

# Find the max average prediction score and return the index + 1 which is the best K
best_index = argmax(averages_array) + 1

print("The best K is: %d" % best_index)

# For this dataset, the best K seems to be 5. 5 nearest neighbors seems to be reasonable
# since it is not to small and not too large, thus our bias and variance are both 
# decently low. 

# Splits the 2 arrays (x, y) into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# Fit the training data
classifier.fit(x_train, y_train)

# Make predicition using classifier and test data
predicition = classifier.predict(x_test, best_index)

# Validation
print (accuracy_score(predicition, y_test))
