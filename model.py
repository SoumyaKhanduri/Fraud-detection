import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import datetime
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

os.chdir('C:/Users/soumya/Desktop/BI/Soumya - Projects')
df = pd.read_csv('Fraud Detection.csv')

df['partner_id'] = df['partner_id'].astype('object')
df['partner_pricing_category'] = df['partner_pricing_category'].astype('object')
df['user_id'] = df['user_id'].astype('object')
df['transaction_number'] = df['transaction_number'].astype('object')
df['transaction_initiation'] = pd.to_datetime(df['transaction_initiation'], infer_datetime_format=True)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for i in range(0,df.shape[1]):
    if df.dtypes[i]=='object':
        df[df.columns[i]] = le.fit_transform(df[df.columns[i]])


from sklearn.model_selection import train_test_split
df['Fraud_Label'] = ['fraud' if x == 1 else 'Not_fraud' for x in df['is_fraud']]
X = df.drop(['is_fraud','transaction_number','user_id','transaction_initiation','country','Fraud_Label'], axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23, stratify = y)


# Fitting Multiple Linear Regression to the Training set
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# Predicting the Test set results
y_pred = rf_model.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)


f = open('Fraud_deploy.pkl', 'wb')
pickle.dump(rf_model, f)
f.close()
