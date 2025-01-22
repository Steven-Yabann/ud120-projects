#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

'''
    C - controls the tradeoff between smooth decision boundary and classifying training points correctly
      - the smaller the C, the larger the margin. The larger the C, the smaller the margin 
    gamma - defines how far the influence of a single training example reaches
          - low value considers far values while high value only considers close values only
'''
    
import sys
from time import time
sys.path.append("./tools/")
from email_preprocess import preprocess
import numpy as np
import collections


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# Import SVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Instantiate the classifier
clf = SVC(kernel='rbf', C=10000)

# Fit the model
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Predict
t0 = time()
pred = clf.predict(features_test)
count = collections.Counter(pred)
print("Predicting Time:", round(time()-t0, 3), "s")
print(count)

# # Test the accuracy
# acc = accuracy_score(pred, labels_test)
# print(f'accuracy: {acc}')
#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one

Answer with 'linear' SVC
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 33.307 s
Predicting Time: 2.695 s
Accuracy: 0.9840728100113766

Answer with 'rbf' SVC
No. of Chris training emails :  7936
No. of Sara training emails :  7884
Training Time: 0.016 s
Predicting Time: 0.339 s
accuracy: 0.8953356086461889
'''



#########################################################
