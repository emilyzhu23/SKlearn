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
def heatMap(df):
    # graphing relations between variables
    corr = df.corr()
    sns.heatmap(corr, xticklabels=df.columns, yticklabels=df.columns)
    plt.show()

def ifBinary(colName, df):
    uniqueList = df[colName].unique()
    uniqueList.sort()
    if len(uniqueList) == 2 and (0 in uniqueList) and (1 in uniqueList):
        return True
    else:
        return False

def groupBinary(vname, df):
    grouped = df.groupby([vname]) #binaryv2 = df.DEATH_EVENT, fix when verifying 
    df_zero = grouped.get_group(0)
    df_one = grouped.get_group(1)
    return df_zero, df_one

def graphBinaryBarGraph(vname1, binaryNameY, df, binsNum):
    # vname1 is string
    grouped = df.groupby([binaryNameY]) #binaryv2 = df.DEATH_EVENT, fix when verifying 
    df_zero = grouped.get_group(0)
    df_one = grouped.get_group(1)
    plt.figure()
    plt.hist([df_zero[vname1],df_one[vname1]], bins = binsNum, stacked=True, density=True)
    plt.legend(("0", "1"), loc='upper right')
    plt.xlabel(vname1)
    plt.show()
    # add some prediction stuff

def graphScatterPlot(v1, v2):
    plt.scatter(v1, v2)
    plt.show()

def graphRelation(vname1, vname2, df):
    bin1 = ifBinary(vname1, df)
    bin2 = ifBinary(vname2, df)
    if bin1 and not bin2: # x is binary and y isn't
        X = df[vname2]
        y = df[vname1]
    elif bin1 and bin2:
        graphBinaryBarGraph(vname1, vname2, df, 2)
        return
    else:
        X = df[vname1]
        y = df[vname2]
    
    graphScatterPlot(X, y)

"""
# pressure and death
grouped = df.groupby(df.DEATH_EVENT)
df_zero = grouped.get_group(0)
df_one = grouped.get_group(1)
plt.figure()
plt.hist([df_zero["high_blood_pressure"],df_one["high_blood_pressure"]], bins = 15, stacked=True, density=True)
plt.legend(('Death', 'No Death'), loc='upper right')
plt.xlabel("high_blood_pressure")
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
plt.show()
"""
# ---------------------------------------------
def logRegrCalcAccuracy(X_train, X_test, y_train, y_test):
# logistic regression w/ minmax
    logRegr = LogisticRegression().fit(X_train, y_train)

    predictLogY = logRegr.predict(X_test)

    accuracyscore = accuracy_score(y_test, predictLogY)
    print("logisticregr accuracy score:")
    print(accuracyscore)

# ---------------------------------------------
def ridgeClassCalcAccuracy(X_train, X_test, y_train, y_test):
    ridgeC = RidgeClassifier().fit(X_train, y_train)

    predictRidgeY = ridgeC.predict(X_test)

    accuracyscore = accuracy_score(y_test, predictRidgeY)
    print("ridgeclass:")
    print(accuracyscore)

# ---------------------------------------------
def main():
    # ---------------------------------------------
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    # no scaling
    X = df[df.columns[0:12]]
    y = df[df.columns[12]]
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
    print("minmaxscaler ----")
    ridgeClassCalcAccuracy(X_train_mm, X_test_mm, y_train_mm, y_test_mm)
    logRegrCalcAccuracy(X_train_mm, X_test_mm, y_train_mm, y_test_mm)
    print("maxabs ----")
    ridgeClassCalcAccuracy(X_train_ma, X_test_ma, y_train_ma, y_test_ma)
    logRegrCalcAccuracy(X_train_ma, X_test_ma, y_train_ma, y_test_ma)
    # ---------------------------------------------
    heatMap(df)
    # ---------------------------------------------
    quit = False
    while quit == False:
        userInput = input("Graph relationship between 2 variables - Format: X variable x Y variable: ")
        if userInput == "quit":
            quit = True
        else:
            vars = userInput.split(" x ")
            graphRelation(vars[0], vars[1], df)
main()
    