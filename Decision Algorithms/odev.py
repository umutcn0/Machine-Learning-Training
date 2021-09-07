import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score
df = pd.read_csv("maaslar_yeni.csv")

x = df.iloc[:,2:5]
y = df.iloc[:,5:]
X = x.values
Y = y.values

print(df.corr())
# MULTİPLE LİNEAR REGRESSİON
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X,Y)
print(lin_reg.predict(X))
#P VALUE VE R2 DEĞERİ OLS MODELİ
print("MLP")
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
#R2 DEĞERİ
print(r2_score(Y,lin_reg.predict(X)))

#POLYNOMİAL REGRESSİON
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures()
x_poly = poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(x_poly,y)
print("POLYNOMİAL")
model = sm.OLS(lin_reg2.predict(x_poly),x)
print(model.fit().summary())
print(r2_score(y,lin_reg2.predict(x_poly)))

#SVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_scale = sc.fit_transform(x)
sc2 = StandardScaler()
y_scale = sc.fit_transform(y)

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_scale,y_scale)
print("SVR")
model = sm.OLS(svr_reg.predict(x_scale),x_scale)
print(model.fit().summary())
print(r2_score(y_scale,svr_reg.predict(x_scale)))

#DESİCİON TREE
from sklearn.tree import DecisionTreeRegressor

des_reg = DecisionTreeRegressor(random_state=0)
des_reg.fit(x,y)
print("DESİCİON")
model = sm.OLS(des_reg.predict(x),x)
print(model.fit().summary())
print(r2_score(y,des_reg.predict(x)))

#RANDOM FOREST
from sklearn.ensemble import RandomForestRegressor
rand_reg = RandomForestRegressor(n_estimators=10,random_state=42)
rand_reg.fit(x,y)

print("RANDOM FOREST")
model = sm.OLS(rand_reg.predict(x),x)
print(model.fit().summary())
print(r2_score(y,rand_reg.predict(x)))























