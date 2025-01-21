#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


##############################################################
# Enter Your Code Here
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create the classifier
clf = GaussianNB()

# Fit the classifier
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Predict using the test 
t0 = time()
prediction = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(prediction, labels_test)

print(f'accuracy: {accuracy}')

##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

'''
.venvstevenyabann@Stevens-MacBook-Pro ud120-projects % python3 naive_bayes/nb_author_id.py
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 0.325 s
Predicting Time: 0.031 s
accuracy: $0.9732650739476678
'''
##############################################################