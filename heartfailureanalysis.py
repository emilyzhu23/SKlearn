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
"""
Purpose: Draw a heatmap with all of the variables from the dataset
Parameters: df, pandas dataframe of all data from csv 
Return: none
"""
    # graphing relations between variables
    corr = df.corr()
    sns.heatmap(corr, xticklabels=df.columns, yticklabels=df.columns)
    plt.show()

def ifBinary(colName, df):
"""
Purpose: Determine if variable is binary
Parameters: colName, str - name of variable; df, pandas dataframe of all data from csv 
Return: True/False, boolean - if variable is binary
"""
    uniqueList = df[colName].unique()
    uniqueList.sort()
    if len(uniqueList) == 2 and (0 in uniqueList) and (1 in uniqueList):
        return True
    else:
        return False

def groupBinary(vname, df):
"""
Purpose: Split binary values into 0 and 1
Parameters: vname, str - name of variable; df, pandas dataframe of all data from csv 
Return: df_zero, df_one - groupby objects from pandas dataframe of 0s and 1s separated
"""
    grouped = df.groupby([vname])
    df_zero = grouped.get_group(0)
    df_one = grouped.get_group(1)
    return df_zero, df_one

def graphBinaryBarGraph(vname1, binaryNameY, df, binsNum):
"""
Purpose: Graph binary bar graph
Parameters: vname1, str - name of x variable; binaryNameY, str - name of y variable; df, pandas dataframe of all data from csv; binsNum - int, number of bins to graph
Return: None
"""
    # vname1 is string
    grouped = df.groupby([binaryNameY])
    df_zero = grouped.get_group(0)
    df_one = grouped.get_group(1)
    plt.figure()
    plt.hist([df_zero[vname1],df_one[vname1]], bins = binsNum, stacked=True, density=True)
    plt.legend(("0", "1"), loc='upper right')
    plt.xlabel(vname1)
    plt.show()

def graphScatterPlot(v1, v2):
"""
Purpose: Graph scatter plot
Parameters: v1 - Series object, all values of x variable; v2 - Series object, all values of y variable
Return: None
"""
    plt.scatter(v1, v2)
    V1_train,V1_test,V2_train,V2_test = train_test_split(v1, v2, test_size=0.2)
    logRegr = LogisticRegression().fit(V1_train.values.reshape(-1, 1), V2_train)
    x = np.arange(np.ptp(v1.values, axis = 0))
    print(x)
    plt.plot(x.reshape(-1, 1), logRegr.predict(x.reshape(-1, 1)), color = "green")
    plt.show()

def graphRelation(vname1, vname2, df):
"""
Purpose: Determines how best to graph relationship between certain variables 
Parameters: vname1, str - name of one variable; vname2, str - name of other variable; df, pandas dataframe of all data from csv 
Return: None
"""
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
# ---------------------------------------------
def logRegrCalcAccuracy(X_train, X_test, y_train, y_test):
"""
Purpose: Calculate the accuracy score of the logistic regression equation
Parameters: X_train - pandas dataframe, train x variable values ; X_test  - pandas dataframe, test x variable values;
 y_train  - pandas dataframe, train y variable values; y_test - pandas dataframe, test y variable values 
Return: None
"""
# logistic regression w/ minmax
    logRegr = LogisticRegression().fit(X_train, y_train)

    predictLogY = logRegr.predict(X_test)

    accuracyscore = accuracy_score(y_test, predictLogY)
    print("logisticregr accuracy score:")
    print(accuracyscore)

# ---------------------------------------------
def ridgeClassCalcAccuracy(X_train, X_test, y_train, y_test):
"""
Purpose: Calculate the accuracy score of the ridge classification equation
Parameters: X_train - pandas dataframe, train x variable values ; X_test  - pandas dataframe, test x variable values;
 y_train  - pandas dataframe, train y variable values; y_test - pandas dataframe, test y variable values 
Return: None
"""
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
        print("To quit, type 'quit'")
        userInput = input("Graph relationship between 2 variables - Format: X variable x Y variable: ")
        if userInput == "quit":
            quit = True
        else:
            vars = userInput.split(" x ")
            graphRelation(vars[0], vars[1], df)
main()
    