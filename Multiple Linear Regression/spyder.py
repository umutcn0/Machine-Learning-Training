import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("odev_tenis.csv")

from sklearn import preprocessing
df2 = df[["windy","play"]].apply(preprocessing.LabelEncoder().fit_transform)
df= df.drop(["windy","play"], axis=1)
df = pd.concat([df,df2],axis=1)
col = df[["outlook"]]
ohe = preprocessing.OneHotEncoder()
col = ohe.fit_transform(col).toarray()
col = pd.DataFrame(col)

df= df.drop(["outlook"], axis=1)
df= pd.concat([col,df],axis=1)

y = df["humidity"]
df= df.drop("humidity",axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(df,y,test_size=0.33, random_state=42)

lr = LinearRegression()
lr.fit(x_train,y_train)
result = lr.predict(x_test)
print(result)
import statsmodels.api as sm
ones = np.ones((14,1))
df["b"]= ones

x_l = df.iloc[:,[0,1,2,3,5]].values
x_l = np.array(x_l,dtype=float)
model = sm.OLS(y,x_l)
sonuc = model.fit()
print(sonuc.summary())
























