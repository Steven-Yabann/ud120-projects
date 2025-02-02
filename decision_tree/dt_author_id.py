#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("./tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# print(len(features_test))


#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# initialize the classifier
clf = DecisionTreeClassifier(min_samples_split=40)
# fit the model
clf.fit(features_train, labels_train)
# predict with the model
pred = clf.predict(features_test)
# test the accuracy if* needed
acc = accuracy_score(pred, labels_test)
print(f'the accuracy is {acc}')
#########################################################
'''
the accuracy is 0.9778156996587031
'''


