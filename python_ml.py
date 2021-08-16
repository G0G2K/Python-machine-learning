#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

"""
use of both supervised and unsupervised ml models
to predict whether an institution is Private or 
Public based on other features
"""

# read data file
data = pd.read_csv('ForbesData2019.csv')
print("Data Read OK!\n")

# data cleaning
print("Cleaning data...\n")
# remove object datatype columns
data = data.drop(['Name', 'City', 'State', 'Website'], axis=1)

# converter function turns private 1 public 0
def converter(private):
    if private == 'Private':
        return 1
    else:
        return 0
    
# create new column Cluster with converter function
data['Cluster'] = data['Public/Private'].apply(converter) 

# remove Public/Private and Cluster columns
cleaned_data = data.drop(['Public/Private', 'Cluster'], axis=1)

# remove null values
cleaned_data.dropna(inplace=True)

# unsupervided learning
print("Unsupervised Learning:\n")
# KMeans 
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

kmeans = KMeans(n_clusters=2)
kmeans.fit(cleaned_data)

# remove null values from test data
data.dropna(inplace=True)

# show accuracy by compairing kmeans and original data
print('KMeans report:')
print(classification_report(data['Cluster'], kmeans.labels_))

# hierachical classification
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit(cleaned_data)
print("\nHierachical Classification Report:")
print(classification_report(data['Cluster'], cluster.labels_))

# supervised learning
print("\nSupervised learning models:")
from sklearn.model_selection import train_test_split

# split data for training and testing
X = data.drop(['Public/Private', 'Cluster'], axis=1)
y = data['Cluster']
X_test, X_train, y_test, y_train = train_test_split(X, y)

# decision trees
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

#train model
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print('\nDecision Trees Report:')
print(classification_report(y_test,predictions))

# logistic regression
from sklearn.linear_model import LogisticRegression

# train model
logistic_r = LogisticRegression(max_iter=300).fit(X_train, y_train)
predictions = logistic_r.predict(X_test)
print('\nLogistic Regression:')
print(classification_report(y_test, predictions))


# In[ ]:




