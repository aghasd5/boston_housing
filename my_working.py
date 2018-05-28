# -*- coding: utf-8 -*-
"""
Created on Thu May 17 01:26:23 2018

@author: Aghasd5
"""

#%%

# Import libraries necessary for this project
import numpy as np
import pandas as pd
#from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs



# Pretty display for notebooks
#%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('c://Udacity/projects/boston_housing/housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

""" describes different stats for the data"""
data.describe()

data["RM"].value_counts()

""" 
The info() method is useful to get a quick description of the data,
in particular the total number of rows, and each attriubute's type and number of non-null values ()
"""
data.info()

"""
see pg 44 for the Orielly book to read the TAR File.
"""


#%%

# Implementation: Calculate Statistics

# TODO: Minimum price of the data
minimum_price = prices.min()


# TODO: Maximum price of the data
maximum_price =prices.max()

# TODO: Mean price of the data
mean_price = prices.mean()

# TODO: Median price of the data
median_price = prices.median()

# TODO: Standard deviation of prices of the data
std_price = prices.std()


# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))


#%%

#Implementation: Define a Performance Metric

# TODO: Import 'r2_score'

from sklearn.metrics import r2_score


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


#%%
    
#Question 2 - Goodness of Fit

#Assume that a dataset contains five data points and a model made the following predictions for the target variable:

#True Value - Prediction
#3.0 == 2.5
#-0.5 == 0.0
#2.0 == 2.1
#7.0 == 7.8
#4.2 == 5.3

   
# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))



#%%

#Implementation: Shuffle and Split Data
#Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.

#For the code cell below, you will need to implement the following:
#Use train_test_split from sklearn.model_selection to shuffle and split the features and prices data into training and testing sets.
#Split the data into 80% training and 20% testing.
#Set the random_state for train_test_split to a value of your choice. This ensures results are consistent.
#Assign the train and testing splits to X_train, X_test, y_train, and y_test.

# TODO: Import 'train_test_split'
from sklearn.model_selection import train_test_split

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size=0.2, random_state=1)

# Success
print("Training and testing split was successful.")

""" 
using the Visuals.py class to see the Models with different Max_depth paramter.
"""
#vs.ModelLearning(X_train,y_train)
#

"""
using the 
"""
#vs.ModelComplexity(X_train,y_train)


#%%

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.tree import DecisionTreeRegressor as DTRegressor
from sklearn.metrics import make_scorer as m_scorer
from sklearn.model_selection import GridSearchCV as GCV


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits= 10, test_size = 0.20, random_state = 0)
    
    # TODO: Create a decision tree regressor object
    regressor = DTRegressor(random_state=0)
    #regressor = None

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    dt_range = range(1,11)
    params = dict(max_depth=dt_range)

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = m_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    new_GCV=GCV(regressor,params,scoring= scoring_fnc,cv=cv_sets)
    grid = new_GCV

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


#%%
#Q8 answers below

#**Answer: ** 
#1) It randomly splits the data into K amount of smaller data sets also known as folds.Then testing and evaluating the model on K different folds or datasets and providing the final mean of all k number of runs. 
#
#2) It benefits us to narrow down on the final selection of the hyperparameters.It provides us with an approximation in reference to the model with the best score. In this case, higher the score the better your model is trained.

#%%

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


#%%
#Question 9 - Optimal Model
#What maximum depth does the optimal model have? How does this result compare to your guess in Question 6? 


#%%
# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

#df=pd.dataclient_data
#reg.predict(client_data)

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print ("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
    
#%%

# Sensitivity
    
vs.PredictTrials(features, prices, fit_model, client_data)

