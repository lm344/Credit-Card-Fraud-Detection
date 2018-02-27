import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import scipy.optimize as op
creditcard = pd.read_csv('creditcard.csv')

Independent_features = creditcard[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
Dependent_Features = creditcard['Class']

#Splitting Train and test dataset using sklearn
Xtrain, Xtest, ytrain, ytest = train_test_split(Independent_features, Dependent_Features, test_size=0.65, random_state=0)
print(Xtrain.shape);
print(Xtest.shape);

#call the classifier and train the data
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(penalty='l2');
clf_logistic.fit(Xtrain, ytrain);

# make predition and check accuracy
print("Logistic Regression Output")
ypred = clf_logistic.predict(Xtest);
print(metrics.confusion_matrix(ytest,ypred));
print(metrics.classification_report(ytest,ypred));
print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));
print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));

#checking Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
clf_rf.fit(Xtrain,ytrain)

print("Random Forest Classifier Output")
ypred = clf_rf.predict(Xtest);
print(metrics.confusion_matrix(ytest,ypred));
print(metrics.classification_report(ytest,ypred));
print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));
print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));

#Random Forrest seems to be  pretty good.
