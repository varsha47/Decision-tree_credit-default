# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 01:43:03 2019

@author: Varsha Choudhary
title: "Deafult credit analysis using Decision tree/ Random forest"
date: "January 18, 2019"
"""
#importing libraries for EDA and classification
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


#import bank marketing csv file and bring it to right format
df = pd.read_excel("C:/Users/Varsha/Downloads/default of credit card clients.xls", skiprows=1)
df.head(2)
df.shape
df.columns
df= df.drop(['ID', ], axis=1)
df.shape
df.describe()
cols = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE']
for col in cols:
    print(df[col].unique())
#define labels and faetures
X = np.array(df.drop(['default payment next month'], 1))
y = np.array(df['default payment next month'])

#split train and test data in 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

#scale train data to normal form (mean=0, SD=1)
scX = StandardScaler()
X_train = scX.fit_transform( X_train )
X_test = scX.transform( X_test )

#train data usinf random forest as classifier
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit( X_train, y_train )
y_pred = classifier.predict( X_test )

#create confusion matrix to calculate accuracy of classifier
cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for RandomForest = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresRF = cross_val_score( classifier, X_train, y_train, cv=10)
print("Mean RandomForest CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresRF.mean(), scoresRF.std() ))
