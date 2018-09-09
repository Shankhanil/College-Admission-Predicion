# The PROBLEM STATEMENT : CREATE A SCRIPT TO PREDICT THE CHANCES OF ADMIT TO AN UNIVERSITY BASED ON 
#                         THE ACADEMIC PERFORMANCE, STRENGTH OF RECOMMENDATION AND RESEARCH

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For data visualisation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# read the data from CSV

data = pd.read_csv ("../input/Admission_Predict.csv")

print(data.columns)

# Decide the features to work with

feature_X = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR ', 'CGPA']
feature_Y = ['Chance of Admit ']

data_X = data[feature_X]
data_Y = data[feature_Y]

# Segregate the dataset into TRAIN DATA and TEST DATA

train_X = data_X[0:320]
train_Y = data_Y[0:320]

test_X = data_X[321:399]
test_Y = data_Y[321:399]

# Normalize the featuesre

X_mean = np.mean(train_X)
X_std = np.std(train_X)
train_X_norm = (train_X - X_mean)/X_std
test_X_norm = (test_X - X_mean)/X_std

# Visualise the data

#   These data features have a clear relationship between themseleves and CHANCE OF ADMIT
plt.scatter(train_X_norm['CGPA'], train_Y)
plt.scatter(train_X_norm['GRE Score'], train_Y)
#   These data features have an ambiguous relationship between themseleves and CHANCE OF ADMIT

plt.scatter(train_X_norm['TOEFL Score'], train_Y)             # lightly ambiguous
plt.scatter(train_X_norm['LOR '], train_Y)                    # quite ambiguous
# the rest of the data features have been found to be having little or no effect on the CHANCE OF ADMIT

# Train the predictive mode : LINEAR REGRESSION

regression = LinearRegression ()

regression.fit(train_X_norm, train_Y)

decision_tree.fit(train_X_norm, train_Y)

# Predict data
sample_prediction = regression.predict(test_X_norm.head())

print (sample_prediction)
print (test_Y.head())

# checking accuracy of the data : LINEAR REGRESSION
prediction = regression.predict(test_X_norm)
S_y = np.sqrt(np.sum( (test_Y - prediction)**2 )/len(test_Y))
print(S_y[0] * 100)
#   The error with LINEAR REGRESSION IS AT 6.62%

# Comparing the accuracy of different prediction models

