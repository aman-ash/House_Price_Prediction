from django.shortcuts import render
from django.shortcuts import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    df = pd.read_csv('House_Price/data.csv')
    X = df.drop(['date', 'price', 'yr_built', 'yr_renovated',
                'street', 'city', 'statezip', 'country', 'sqft_above'], axis=1)
    Y = df['price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])

    pred = model.predict(
        np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9]).reshape(1, -1))
    pred = round(pred[0])
    price = 'The predicted price is $ ' + str(pred)

    return render(request, 'predict.html', {"result2": price})
