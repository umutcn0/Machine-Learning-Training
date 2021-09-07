import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("maaslar.csv")

xdf = df.iloc[:,1:2]
ydf = df.iloc[:,2:]

plt.scatter(xdf,ydf)