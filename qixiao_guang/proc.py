# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def cal_mean(c):
    mean_10days=np.zeros(len(c))
    n=len(c)/10
    m=len(c)%10
    s=c[-m:]
    mean_10days[-m:]=np.mean(s).round(6)
    for i in range(n):
        s=c[i*10:(i+1)*10]
        mean_10days[i*10:(i+1)*10]=np.mean(s).round(6)
    return mean_10days

def cal_std(c):
    std_10days=np.zeros(len(c))
    n=len(c)/10
    m=len(c)%10
    s=c[-m:]
    std_10days[-m:]=np.std(s).round(6)
    for i in range(n):
        s=c[i*10:(i+1)*10]
        std_10days[i*10:(i+1)*10]=np.std(s).round(6)
    return std_10days

def cal_var(c):
    var_10days=np.zeros(len(c))
    n=len(c)/10
    m=len(c)%10
    s=c[-m:]
    var_10days[-m:]=np.var(s).round(6)
    for i in range(n):
        s=c[i*10:(i+1)*10]
        var_10days[i*10:(i+1)*10]=np.var(s).round(6)
    return var_10days

def cal_label(c,mean,std):
    label_10days=[]
    for i in range(len(c)):
        if c[i]>=mean[i]-5*std[i] and c[i]<=mean[i]+5*std[i]:
            label_10days.append("medium")
        elif c[i]<mean[i]-5*std[i]:
            label_10days.append("low")
        else:
            label_10days.append("high")
    return label_10days

def cal_MaximumYield(h,l):
    return ((np.abs(h-l))/l).round(6)

def cal_CurrentEarningsMultiples(c):
    mul=np.zeros(len(c))
    for i in range(10,len(c)):
        mul[i]=(c[i]/c[i-10]).round(6)
    return mul

def cal_GrowthPrice(c):
    diff=np.zeros(len(c))
    diff1=np.diff(c)
    diff[1:]=diff1
    return diff

filename=raw_input()
h,l,c=np.loadtxt(filename,delimiter=',',usecols=(2,3,5),unpack=True)
mean_col=cal_mean(c)
std_col=cal_std(c)
var_col=cal_var(c)
label_col=cal_label(c,mean_col,std_col)

MaxY_col=cal_MaximumYield(h,l)
CEM_col=cal_CurrentEarningsMultiples(c)
diff=cal_GrowthPrice(c)
new_data=np.column_stack((c,mean_col,std_col,var_col,MaxY_col,CEM_col,diff,label_col))
dataframe=pd.DataFrame(new_data,columns=['Close price','Mean_10days','Std_10days','Var_10days','MY','CEM','diff','label_10days'])
dataframe.to_csv('new_data.csv')

Y=np.zeros((len(c),),dtype='bool')

X=new_data[:,0:7]
for i in range(len(c)):
    Y[i]=new_data[i][-1]=='medium'

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=24)
estimator=KNeighborsClassifier()
estimator.fit(X_train,Y_train)
y_predict=estimator.predict(X_test)
accuracy=(np.mean(Y_test==y_predict)*100).round(2)
print accuracy







