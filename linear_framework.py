# -*- coding: utf-8 -*-
"""Copy of sklearn lin reg diabetes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZhpR8IY7b-qBSKKtUP-LyYnkZ2VTeitO
"""

from google.colab import drive

drive.mount("/content/gdrive")  
# !pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/gdrive/MyDrive/Project AI Real State"

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


columns = ["No", "Transaction Date", "House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude","Price UPA"]
df = pd.read_csv('./RealEstate.csv',names = columns)

print(df.head())
cleaned_df = df[["House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude","Price UPA"]]

print(cleaned_df.head())

real_state_train, real_state_test = train_test_split(cleaned_df, test_size=0.1, train_size=0.9, shuffle=True)

print("train")
real_state_train.head()
print(len(real_state_train))
print("test")
real_state_test.head()
print(len(real_state_test))

x_real_state_train = real_state_train[["House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude"]]
x_real_state_test = real_state_test[["House Age", "Distance to MRT station", "Number Convinience Stores","Latitude","Longitude"]]

y_real_state_train = real_state_train["Price UPA"]
y_real_state_test = real_state_test["Price UPA"]


regr = linear_model.LinearRegression()


regr.fit(x_real_state_train, y_real_state_train)


real_state_y_pred = regr.predict(x_real_state_test)


print('Coefficients: \n', regr.coef_)

print('Mean squared error: %.2f', mean_squared_error(y_real_state_test, real_state_y_pred))

print('Coefficient of determination: %.2f', r2_score(y_real_state_test, real_state_y_pred))



plt.scatter(x_real_state_test["House Age"], y_real_state_test,  color='black')
plt.plot(x_real_state_test["House Age"],real_state_y_pred , color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()

"""# Input"""

print("Enter the following in order separated by spaces")
print("House Age, Distance to nearest station, Number of Convinience Stores, Latitude and Longitude")
query = list(map(float,input().split()))
query = np.array(query)
user_prediction = regr.predict(query.reshape(1,-1))
print(user_prediction)