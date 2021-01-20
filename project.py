import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import quandl

quandl.ApiConfig.api_key = "ty-JJLdJcMjfmGq1KetK"


df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
labelz = "Adj. Close"
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df["label"] = df[labelz].shift(-forecast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) # features

y = np.array(df['label']) # label > Adj . Close

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)