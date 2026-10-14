# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('food_truck.txt', header = None, delimiter = ",") #read from dataset
X = data.iloc[:,0] # read first column, will be put in a 'series' variable
print('X.shape: ', X.shape)
y = data.iloc[:,1] # read second column, will be put in a 'series' variable
print('y.shape: ', y.shape)
m = len(y) # number of training example (97)
print('Number of samples:', m)
print(data.head()) # view first few rows of the data


print(type(X)) # should be a pd.Series()
X = X.to_numpy()[:,np.newaxis] # convert pd.Series() to an np.ndarray
y = y.to_numpy()[:,np.newaxis]

theta = np.zeros([2,1]) # start off with a (0, 0) array
iterations = 100000 #
learning_rate = 0.01 # magic number, defines the increase in change in Theta in every iteration we take
X_ones = np.ones([m,1])
X = np.hstack((X_ones, X))

def computeCost(y, theta):
    # We are going to calculate the MSE (Mean Square Error). NOT the RMSE (Root Mean Square Error)

    temp = np.dot(X, theta) - y
    # FILL IN THE NECESSARY CODE. Return the cost.
    return np.sum(np.power(temp, 2)) / (2*len(y))

cost = computeCost(y, theta)
print(cost)

def gradientDescent(X_ones, y, theta, learning_rate, iterations):
    # find the partial derivative of the cost
    for _ in range(iterations):
        iterat = X.dot(theta)
        error = iterat - y

        gradient = 2*X.T.dot(error)/len(X)

        theta -= learning_rate * gradient


    # First work out what the partial derivatives are, then apply the gradientDescent
    # Parameters:
    # -----------
    # X: matrix with the input variables
    # y: the correct results
    # iterations: The number of times the gradientDescent should run.
    #
    # FILL IN THE NECESSARY CODE.
    # After the iterations are performed, return the theta0 and theta1 values.
    #
    #

    return theta

theta_refined = gradientDescent(X_ones, y, theta,0.01, 7500)
print(computeCost(y, theta_refined))

#Error default iterations 4.477
#
