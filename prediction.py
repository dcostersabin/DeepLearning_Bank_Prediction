#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:00:29 2020

@author: dcoster
"""

import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV


# --------------------------data pre processing---------------------- 
data_file = "Datasets/Data.csv"
dataset = pd.read_csv(data_file)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
transformer = ColumnTransformer([("Country", OneHotEncoder(dtype=np.float64), [1],)], remainder='passthrough')
X = transformer.fit_transform(X)
X = X[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# --------------------data processing ends-----------------------


# ---------------------neural network starts---------------------------
# classifier = Sequential()
# classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
# classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
# classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# classifier.fit(X_train, y_train, batch_size=10, epochs=100)
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred[0] > 0.5)
# new_data = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
# new_prediction = classifier.predict(sc.transform(new_data))
# new_prediction = (new_prediction[0] > 0.5)
# ----------------------neural networl ends


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10 , epochs=100)
accuracies = cross_val_score(estimator = classifier , X = X_train , y = y_train , cv = 10, n_jobs = -1 )
mean = accuracies.mean()
vaciance = accuracies.std()

# --------------------------paramater tuning-----------------------

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(p = 0.1))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'epochs':[100,200],
              'optimizer':['adam','rmsprop']
              }
grid_search = GridSearchCV(estimator = classifier ,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv= 10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_









