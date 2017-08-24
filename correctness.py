import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import datetime

df_train = pd.read_csv('file_name', delimiter="|")
df_test = pd.read_csv('file_name', delimiter="|")
