from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Using Librariry

h = LinearRegression()  # create the model
# 1.1 Features (Xi)

X = [[230.1, 37.8, 69.2], [44.5, 39.3, 45.1], [17.2, 45.9, 69.3],

     [151.5, 41.3, 58.5], [180.8, 10.8, 58.4], [8.7, 48.9, 75], [57.5, 32.8, 23.5],

     [120.2, 19.6, 11.6], [8.6, 2.1, 1], [199.8, 2.6, 21.2]]

# 1.2 Target (actual_y)

y = [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6]
H = LinearRegression()
H.fit(X, y)
print(H.intercept_, H.coef_)

y_predicted=H.predict([[230.1, 37.8, 69.2]])

mse=metrics.mean_squared_error([22.1],y_predicted)



#From scratch
X1=np.array([1,2,3,4])
y1=np.array([0.1,0.4,0.8,1.2])


def predict(X1,w,bias):
    return X1*w+bias

def fit(X1,y1):
    n=len(X1)
    bias=0
    epochs=1000
    w=0
    lr=0.0001
    for epoch in range(epochs):
        y_predicted=predict(X1,w,bias)
        mse=(1/n)*np.sum((y1-y_predicted)**2)
        dw=-(2/n)*np.sum(X1*(y1-y_predicted))
        dbias=(2/n)*np.sum(y1-y_predicted)
        if(epochs%5==0):
            plt.scatter(data_Hours,data_Scores)
            plt.plot(data_Hours,w*np.array(data_Hours)+bias)
            plt.show()

        w=w-lr*dw
        bias=bias-lr*dbias
    return w,bias


data=pd.read_csv("student_scores.csv")
data_Hours=[]
data_Scores=[]

for i in range(data.__len__()):
    data_Hours.append(data.iloc[i].Hours)
    data_Scores.append(data.iloc[i].Scores)


w,bias=fit(np.array(data_Hours),np.array(data_Scores))



