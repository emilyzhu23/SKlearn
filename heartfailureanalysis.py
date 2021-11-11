import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# ---------------------------------------------
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
# no scaling
X = df[df.columns[0:12]]
y = df[df.columns[12]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# ---------------------------------------------
# graphing relations between variables
corr = df.corr()
sns.heatmap(corr, xticklabels=df.columns, yticklabels=df.columns)

# kinase and death
grouped = df.groupby(df.DEATH_EVENT)
df_zero = grouped.get_group(0)
df_one = grouped.get_group(0)
plt.figure()
plt.hist([df_zero["creatinine_phosphokinase"],df_one["creatinine_phosphokinase"]], bins = 15, stacked=True, density=True)
plt.legend(('Death', 'No Death'), loc='upper right')
plt.xlabel("creatinine_phosphokinase")
# creatinine and death
plt.figure()
plt.hist([df_zero["serum_creatinine"],df_one["serum_creatinine"]], bins = 15, stacked=True, density=True)
plt.legend(('Death', 'No Death'), loc='upper right')
plt.xlabel("serum_creatinine")
# age and death
plt.figure()
plt.hist([df_zero["age"],df_one["age"]], bins = 15, stacked=True, density=True)
plt.legend(('Death', 'No Death'), loc='upper right')
plt.xlabel("age")

# ---------------------------------------------
# scaling w/ minmax
mm_scaler = preprocessing.MinMaxScaler()
X_minmax = mm_scaler.fit_transform(X)

X_train_mm,X_test_mm,y_train_mm,y_test_mm = train_test_split(X_minmax, y, test_size=0.2)

# ---------------------------------------------
# scaling w/ maxabs
MaxAbsScaler = preprocessing.MaxAbsScaler()
X_maxabs = MaxAbsScaler.fit_transform(X)

X_train_ma,X_test_ma,y_train_ma,y_test_ma = train_test_split(X_maxabs, y, test_size=0.2)
# ---------------------------------------------
# logistic regression w/ minmax
logRegr = LogisticRegression().fit(X_train_mm, y_train_mm)

predictLogY = logRegr.predict(X_test_mm)

accuracyscore = accuracy_score(y_test_mm, predictLogY)
print("logisticregr - mm")
print(accuracyscore)

# logistic regression w/ maxabs
logRegr = LogisticRegression().fit(X_train_ma, y_train_ma)
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
