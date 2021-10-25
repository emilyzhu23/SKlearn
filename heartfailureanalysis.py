import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

mm_scaler = preprocessing.MinMaxScaler()
X = df[df.columns[0:12]]
X_minmax = mm_scaler.fit_transform(X)

y = df[df.columns[12]]

X_train,X_test,y_train,y_test = train_test_split(X_minmax, y, test_size=0.2)
print(X_train)

logRegr = LogisticRegression().fit(X_train, y_train)

predictFake = logRegr.predict(X_train)
fakeaccscore = accuracy_score(y_train, predictFake)
print("should be 1")
print(fakeaccscore)

predictLogY = logRegr.predict(X_test)

accuracyscore = accuracy_score(y_test, predictLogY)
print("logisticregr")
print(accuracyscore)
# ---------------------------------------------
ridgeC = RidgeClassifier().fit(X_train, y_train)

predictRidgeY = ridgeC.predict(X_test)

accuracyscore = accuracy_score(y_test, predictRidgeY)
print("ridgeclass")
print(accuracyscore)

# ---------------------------------------------


