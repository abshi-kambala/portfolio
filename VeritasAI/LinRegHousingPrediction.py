
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing
path = "/content/sample_data/california_housing_train.csv" # paste the path here
df = pd.read_csv(path)

'''

X = df['median_income']
y = df['median_house_value']

plt.scatter(X, y)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.show()

'''

X = df.drop(['median_house_value'], axis = 1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test)
y_test = np.array(y_test).reshape(-1,1)

print(X_train.shape, X_test.shape)

lm = LinearRegression()
lm.fit = lm.fit(X_train, y_train)
lm.coef_
RMSE = (mean_squared_error(y_test, lm.predict(X_test)))**(1/2)

RMSE
