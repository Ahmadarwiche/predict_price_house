from django.contrib import messages
from . import forms
from .forms import FLOOR_CHOICES
from django.shortcuts import render, redirect, reverse
import numpy as np
import pandas as pd
#import pytz
#import six
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import MinMaxScaler


def home(request):
     return render(request, "home.html" )


def result(request):
     df=pd.read_csv('prediction/data/kc_house_data.csv')
     df['yr_renovated'] = df[["yr_renovated","yr_built"]].apply( lambda x : x[1] if x[0]==0 else x[0], axis=1) 
     y = df['price']
     X = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_basement','yr_renovated','zipcode','lat','long']]
     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=30)
     ###################################################################################################
     # df.zipcode=df.zipcode.astype('object')
     # categorial_features = X.zipcode
     # numeric_features = df[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_basement','yr_renovated','lat','long']]
     # from sklearn.pipeline import Pipeline
     # numeric_transformer = Pipeline([
     #    ('imputer', SimpleImputer(strategy='mean')),
     #    ('min_max', MinMaxScaler()), 
     #    ])
     # from sklearn.preprocessing import OneHotEncoder
     # categorical_transformer = OneHotEncoder(sparse=True)
     # from sklearn.compose import ColumnTransformer
     # preprocessor = ColumnTransformer(
     # transformers=[
     #           ('num', numeric_transformer, numeric_features),
     #           ('cat', categorical_transformer, categorial_features)
     #      ],
     #      remainder ='passthrough'
     #      )
     # lr = LinearRegression()
     # pipe = Pipeline([
     #           ('prep', preprocessor),
     #           ('lr', lr)
     #      ])
     # trained_pipe = pipe.fit(X_train,y_train)
     ########################################################################################################
     lr = LinearRegression()
     lr.fit(X_train,y_train)
     if request.GET['v3'] and request.GET['v4'] and request.GET['v10'] and request.GET['v11'] and request.GET['v13'] and request.GET['v14'] != ''  :
         var1=float(request.GET['v1'])
         var2=float(request.GET['v2'])
         var3=float(request.GET['v3'])
         var4=float(request.GET['v4'])
         var5=float(request.GET['v5'])
         var6=float(request.GET['v6'])
         var7=float(request.GET['v7'])
         var8=float(request.GET['v8'])
         var9=float(request.GET['v9'])
         var10=float(request.GET['v10'])
         var11=float(request.GET['v11'])
         var12=float(request.GET['v12'])
         var13=float(request.GET['v13'])
         var14=float(request.GET['v14'])
         pred=lr.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14]).reshape(1, -1))
         pred= round(pred[0])
         price = 'The predicted price is : $'+str(pred)
         return render(request, 'home.html',{'result2': price })
     else:
        msg = 'Please enter the missing value !'
        return render(request, 'home.html',{'msg': msg })
   

def Gps(request):
    return redirect('https://www.coordonnees-gps.fr/')
     

def Redirect(request):
     msg1 = 'Please click on Re-predict to restart !'
     return render(request, 'home.html',{'msg1': msg1 })
     
        
