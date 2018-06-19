# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:31:38 2018

@author: 593378
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

filename = ("C:/Users/593378/Karthik/Python/DataScience/SolarSolution/PV_Data_V2.0_Forecasting.csv")

df = pd.read_csv(filename, parse_dates=[['Date', 'Time']])

df1 = df.drop(columns=['Ambient Temperature (degC)', 'Clearsky GHI (W/m2)', 'Simulated Module Wise Power (W)', 'Actual Module wise Power (W)', 'Actual String wise Power (kW)', 'Actual Array wise Power (kW)' ])

independent_variables = ['GHI', 'Modul_ Temperature']

#Multivariant Regression

X = df1[independent_variables]
y = df1['Actual_Plant_Power']

# Split your data set into 80/20 for train/test datasets
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.80, random_state=1)

# create a fitted model 
import pickle
lm = sm.OLS(y_train, X_train).fit()
filename = 'PV_model.pkl'

pickle.dump(lm, open(filename,'wb'))





