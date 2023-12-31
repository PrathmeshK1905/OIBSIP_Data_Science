# -*- coding: utf-8 -*-
"""Email_Spam_Identifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14u1N6ldV2xOZfVW4qO4IqpFXjcgssjgW

Name: Prathmesh Kulkarni

Importing Libraries
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Importing Dataset"""

df = pd.read_csv("/content/spam.csv",encoding="ISO-8859-1")

print(df)

data = df.where((pd.notnull(df)),'')

data.head()

data.info()

data.shape

data.loc[data['v1'] == 'spam','v1',] = 0
data.loc[data['v1'] == 'ham','v1',] = 1

x = data['v2']
y = data['v1']

print(x)

print(y)

"""Splitting Data into Training and Testing"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=3)

print(x.shape)
print(x_train.shape)
print(x_test.shape)

print(y.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english',lowercase = True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_train)

print(x_train_features)

"""Training the model"""

model = LogisticRegression()

model.fit(x_train_features,y_train)

prediction_train = model.predict(x_train_features)
accuracy_train = accuracy_score(y_train, prediction_train)

"""Training Accuracy"""

print("Accuracy on traininng Data:", accuracy_train)

"""Tesing Accuracy"""

prediction_test = model.predict(x_test_features)
accuracy_test = accuracy_score(y_test, prediction_test)

print("Accuracy on Test Data:", accuracy_test)

"""Deployment and Prediction"""

input_your_mail = ["This is the 2nd time we have tried to contact you. You have won the A$400 prize. 2 claim is easy, just call 9876543210"]
input_your_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_your_features)

print(prediction)

if(prediction[0]==1):
  print("Ham mail")
else:
  print("Spam mail")

input_your_mail = ["Wassup! Homie"]
input_your_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_your_features)

print(prediction)

if(prediction[0]==1):
  print("Ham mail")
else:
  print("Spam mail")

"""END"""