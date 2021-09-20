
import pandas as pd

import numpy as np

salary_train=pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/SVM/SalaryData_Train (1).csv")
salary_test=pd.read_csv("C:/Users/Bibhuti/OneDrive/Desktop/360digiTMG assignment/SVM/SalaryData_Test (1).csv")

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

salary_train.columns
salary_test.columns
string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

col_names=list(salary_train.columns)
train_X=salary_train[col_names[0:13]]
train_Y=salary_train[col_names[13]]
test_x=salary_test[col_names[0:13]]
test_y=salary_test[col_names[13]]

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_Y)
pred_test_linear = model_linear.predict(test_x)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_Y)
pred_test_rbf = model_rbf.predict(test_x)

np.mean(pred_test_rbf==test_y)