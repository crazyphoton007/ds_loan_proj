# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:32:28 2020

@author: Shukla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
loans = pd.read_csv('loan_data.csv')
loans.head(2)
loans.info()
loans.describe()
loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy=1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy=0',alpha=0.6)
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy=1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy=0',alpha=0.6)
plt.legend()
plt.xlabel("FICO")
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='Not fully paid=1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label='Not fully paid=0',alpha=0.6)
plt.legend()
plt.xlabel("FICO")
plt.figure(figsize=(15,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=loans,color='black')
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
cat_feats=['purpose']
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()
final_data.head()
from sklearn.model_selection import train_test_split
X=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix 
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier 
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))














