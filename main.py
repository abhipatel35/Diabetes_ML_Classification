# I'll solve a classification ML Problem using complete lifecycle of Ml project.

import pandas as pd

from sklearn.model_selection import train_test_split  # train the data for cleaning and testing part
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# ------------------- Part for data preparation -----------------------------

df = pd.read_csv('diabetes.csv')  # create data frame to import my csv dataset
print(df.head())  # let's see what i have in dataset
print(df.shape)  # no. of rows and columns
print(df.isnull().sum())  # check if have any null values or not.
print(df.info())  # check datatypes of all the data. here all datatypes are numerical values , so we do not have encoding/feature-cleaning for categorical data.

# separate independent feature x and dependent variables y
x = df.drop(columns=['Outcome'], axis=1)
print(x)
y = df['Outcome']
print(y)

# clean the data for train and testing part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 80% data for training and 20% data for testing
print(x_train.shape)
print(x_test.shape)

# --------------------------------------------------------------------------------------

# --------------- Now, let's import a different machine learning model to solve classification problem ------------------
# Since this is classification ML problem , so I will import three different ML models in a classification: Logistic regression, decision tree classifier & random forest classifier.
# Here, I will use Accuracy Matrix as evaluation matrix to compare the accuracy of these three models.

# below steps for train three ML models-------------
# create one instance for Logistic Regression
lr = LogisticRegression()
print(lr.fit(x_train, y_train))
# create one instance for Decision Tree Classifier
dtc = DecisionTreeClassifier()
print(dtc.fit(x_train, y_train))
# create one instance for Random forest Classifier
rfc = RandomForestClassifier()
print(rfc.fit(x_train, y_train))

# Test the Data--------
# store the prediction made by these three different models
lr_pred = lr.predict(x_test)
dtc_pred = dtc.predict(x_test)
rfc_pred = rfc.predict(x_test)

#  to get accuracy of these models that I have trained and based on their prediction
print("Accuracy for logistic regression ->", accuracy_score(y_test, lr_pred))
print("Accuracy for decision tree classifier ->", accuracy_score(y_test, dtc_pred))
print("Accuracy for Random forest classifier ->", accuracy_score(y_test, rfc_pred))

# ---------------------------------------------------------------------------------
