import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import datetime

df_train = pd.read_csv('file_name', delimiter="|")
df_test = pd.read_csv('file_name', delimiter="|")

def processing_data(df):
    df.columns = ["Image_ID", "Actual_Total", "Actual_date", "Actual_time","Predicted_Total","Predicted_date","Predicted_time","conf_total","conf_date","conf_time"]
    columns = ["Diff_total", "Diff_date", "Diff_time", "conf_total", "Actual_result"]
    df1 = pd.DataFrame(columns=columns)
    df1["Diff_total"] = df["Actual_Total"].values - df["Predicted_Total"].values

    df["Actual_date"] = pd.to_datetime(df["Actual_date"])
    df["Predicted_date"] = pd.to_datetime(df["Predicted_date"])
    df1["Diff_date"] = df["Actual_date"].values - df["Predicted_date"].values
    df1["Diff_date"] = df1["Diff_date"].dt.days.astype(int)

    for i in range(0, len(df["Actual_time"]) - 1):
        df.set_value(i, "Actual_time", datetime.datetime.strptime(df["Actual_time"][i], "%H:%M:%S"))
    
    for i in range(0, len(df["Predicted_time"]) - 1):
        if df["Predicted_time"][i] != "noTime":
            df.set_value(i, "Predicted_time", datetime.datetime.strptime(df["Predicted_time"][i], "%H:%M:%S"))
    
    for i in range(0, len(df["Predicted_time"])-1):
        if df["Predicted_time"][i] != "noTime":
            df1.set_value(i, "Diff_time", int((df["Actual_time"][i] - df["Predicted_time"][i]).total_seconds()))
    
    df1["conf_total"] = df["conf_total"] + df["conf_date"] + df["conf_time"]

    for i in range(0,len(df1["Diff_total"])-1):
        if (df1["Diff_total"][i] == 0 and df1["Diff_date"][i] == 0 and df1["Diff_time"][i] == 0):
            df1.set_value(i,"Actual_result",int(1))
        else:
            df1.set_value(i,"Actual_result",int(0))
    
    df1 = df1[~df1.Diff_time.isnull()]
    df1 = df1[~df1.Actual_result.isnull()]
    df1 = df1.astype(int)
    
    return df1

df1_train = processing_data(df_train)
df1_test = processing_data(df_test)

classifier = SVC()
classifier.fit(df1_train.ix[:,:-2], df1_train.ix[:,-1])

predicted_labels = classifier.predict(df1_test.ix[:,:-2])
expected_labels = df1_test.ix[:,-1]
accuracy = classifier.score(df1_test.ix[:,:-2], expected_labels)

print("Accuracy of this Clssifier is : " + str(accuracy))

'''
For this Dataset 3 feature sets were used. This got the accuracy of 99.79%.
'''
