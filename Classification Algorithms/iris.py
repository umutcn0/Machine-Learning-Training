import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score

df= pd.read_csv("Iris.csv")

df.drop("Id",axis=1,inplace=True)
x= df.iloc[:,:4].values
y= df["Species"].values




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.3,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
y_pred = log_reg.predict(x_test)

print("LOG REG")
cm= confusion_matrix(y_pred,y_test)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.svm import SVC
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print("SVC")
cm= confusion_matrix(y_pred,y_test)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("KNN")
cm= confusion_matrix(y_pred,y_test)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


from sklearn.naive_bayes import GaussianNB

nav_b = GaussianNB()
nav_b.fit(x_train,y_train)
y_pred = nav_b.predict(x_test)
print("Gaussian")
cm= confusion_matrix(y_pred,y_test)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(criterion="gini",random_state=42)
dec_tree.fit(x_train,y_train)

y_pred = dec_tree.predict(x_test)

print("Decision Tree")
cm= confusion_matrix(y_pred,y_test)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.ensemble import RandomForestClassifier

rand = RandomForestClassifier(n_estimators= 3,criterion="gini",random_state=42)
rand.fit(x_train,y_train)
y_pred = rand.predict(x_test)
print("Rand Forest")
cm= confusion_matrix(y_pred,y_test)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)




























