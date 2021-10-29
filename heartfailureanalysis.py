import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# graph, try other scaling, regression models
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
# ---------------------------------------------
# graphing
# ---------------------------------------------
# scaling w/ minmax
mm_scaler = preprocessing.MinMaxScaler()
X = df[df.columns[0:12]]
X_minmax = mm_scaler.fit_transform(X)

y = df[df.columns[12]]

X_train_mm,X_test_mm,y_train_mm,y_test_mm = train_test_split(X_minmax, y, test_size=0.2)

# ---------------------------------------------
# scaling w/ maxabs
MaxAbsScaler = preprocessing.MaxAbsScaler()
X = df[df.columns[0:12]]
X_maxabs = MaxAbsScaler.fit_transform(X)

y = df[df.columns[12]]

X_train_ma,X_test_ma,y_train_ma,y_test_ma = train_test_split(X_maxabs, y, test_size=0.2)
# ---------------------------------------------
# K-fold
# ---------------------------------------------
# logistic regression w/ minmax
logRegr = LogisticRegression().fit(X_train_mm, y_train_mm)

predictFake = logRegr.predict(X_train_mm)
predictLogY = logRegr.predict(X_test_mm)

accuracyscore = accuracy_score(y_test_mm, predictLogY)
print("logisticregr - mm")
print(accuracyscore)

# logistic regression w/ maxabs
logRegr = LogisticRegression().fit(X_train_ma, y_train_ma)

predictFake = logRegr.predict(X_train_ma)
fakeaccscore = accuracy_score(y_train_ma, predictFake)
predictLogY = logRegr.predict(X_test_ma)

accuracyscore = accuracy_score(y_test_ma, predictLogY)
print("logisticregr - ma")
print(accuracyscore)
# ---------------------------------------------
# ridge classifier w/ minmax
ridgeC = RidgeClassifier().fit(X_train_mm, y_train_mm)

predictRidgeY = ridgeC.predict(X_test_mm)

accuracyscore = accuracy_score(y_test_mm, predictRidgeY)
print("ridgeclass - mm")
print(accuracyscore)

# ridge classifier w/ maxabs
ridgeC = RidgeClassifier().fit(X_train_ma, y_train_ma)

predictRidgeY = ridgeC.predict(X_test_ma)

accuracyscore = accuracy_score(y_test_ma, predictRidgeY)
print("ridgeclass - ma")
print(accuracyscore)
# ---------------------------------------------



