# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 18:11:02 2019

@author: shivam.k
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset 
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values #returns an array of values for each line
y=dataset.iloc[:, 3].values #returns an array of values for each line and with column index 3 i.e purchased

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values = "NaN", strategy = "mean",axis= 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
#  Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Country column is encoded
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() # to provide dummy values for the country 
# to resolve france > spain and spain > Germany

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)