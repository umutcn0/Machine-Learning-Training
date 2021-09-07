import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')
df = df.dropna(how='any', subset = ['Age','Fare','Sex'])
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
X = df[['Age','Fare','Sex']]
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X , y, test_size=0.3,random_state=0)

# LOGİSTİC REGRESSİON
log_reg = LogisticRegression(random_state=42)
log_reg.fit(x_train,y_train)

y_pred = log_reg.predict(x_test)

cm = confusion_matrix(y_pred,y_test)
print(cm)

accuracy = accuracy_score(y_pred,y_test)
print(accuracy)

dens = log_reg.sparsify()
print(dens.coef_)


# KNN

knn = KNeighborsRegressor(n_neighbors=3,metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("---KNN---")
print(y_pred)

