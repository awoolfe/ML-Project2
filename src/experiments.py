###########################################################
# File name: experiments.py
# Purpose: comparison of sklearn classifiers
###########################################################

from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import sys
import time

X = []
y = []
'''initialize decision tree with max depth (to-do: test optimal max depths)'''
dtree = DecisionTreeClassifier(criterion = "gini", splitter = "best", max_depth = 3)
dtree.fit(X, y)
dtree.predict(X, y)
'''k-means classifier'''
kmeans = KMeans(n_clusters = 20)
kmeans.fit_predict(X,y)

