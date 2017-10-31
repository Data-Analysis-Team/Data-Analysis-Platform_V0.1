from flask import current_app
from app import db
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error
#from sklearn.cross_validation import train_test_split
import neurolab as nl
import neurolab.error as error 
import math
import pymysql

import random
from pyecharts import Scatter3D
from pyecharts.constants import DEFAULT_HOST
from pyecharts import Bar
import random
import scipy.stats as sts

class Developer(db.Model):    
    __tablename__ = 'developers'    
    id = db.Column(db.Integer, primary_key=True)    
    dev_key = db.Column(db.String(40), unique=True, index=True)    
    platform = db.Column(db.String(50))    
    platform_id = db.Column(db.String(40), unique=True)    
    username = db.Column(db.String(150), index=True)    
    integrations = db.relationship('Integration', backref='developer')    
    channels = db.relationship('Channel', backref='developer')

class Integration(db.Model):    
    __tablename__ = 'integrations'    
    id = db.Column(db.Integer, primary_key=True)    
    integration_id = db.Column(db.String(40), unique=True)    
    name = db.Column(db.String(100))    
    description = db.Column(db.String(150))    
    icon = db.Column(db.String(150))    
    channel = db.Column(db.String(150))    
    token = db.Column(db.String(150))    
    developer_id = db.Column(db.Integer, db.ForeignKey('developers.id'))

class Channel(db.Model):    
    __tablename__ = 'channels'    
    id = db.Column(db.Integer, primary_key=True)    
    developer_id = db.Column(db.Integer, db.ForeignKey('developers.id'))    
    channel = db.Column(db.String(150))    

    def __repr__(self):        
        return '<Channel %r>' % self.channel

class Get_Data():
    def get_data(self,db_name,table_name):
         global data
         sql_columns_name=str("desc %s" %table_name)
         db=pymysql.connect("localhost","root","123456",db_name,charset = 'utf8')
         cursor=db.cursor()
         cursor.execute(sql_columns_name)
         columns_name =list(cursor.fetchall())
         sql_data_count=str("select count(*) from %s" %table_name)
         cursor.execute(sql_data_count)
         data_count=list(cursor.fetchall())
         
         sql_data=str("select * from %s" %table_name)
         cursor.execute(sql_data)
         data = pd.DataFrame(np.array(cursor.fetchall()))
         data.columns=[np.array(columns_name)[:,0]]
         db.close()
         return columns_name,data_count,data

class Statistics():
    def descriptive_statistics(self,data):
        des=data.describe().round(2)
        count=des.loc['count']
        mean=des.loc['mean']
        std=des.loc['std']
        mn=des.loc['min']
        per25=des.loc['25%']
        per50=des.loc['50%']
        per75=des.loc['75%']
        mx=des.loc['max']
        #离散系数
        cv=(std/mean).round(2)
        #峰度  
        kurtosis=sts.kurtosis(data).round(2)
        #偏度
        skewness=sts.skew(data).round(2)
        
        return count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness

    def pearson(self,x,y):
        pearson=sts.pearsonr(x,y)
        return pearson
    
    def spearman(self,x,y):
        spearman=sts.spearmanr(x,y)
        return spearman

class PCA():
    def __init__(self,A):
        self.A=A
    def SVDdecompose(self):
        B=np.linalg.svd(self.A,full_matrices=False)
        U=B[0]
        lamda=B[1]
        V=B[2]
        i= len(lamda)
        S=np.zeros((i,i))
        S[:i,:i]=np.diag(lamda)
        self.T=np.dot(U,S)
        V=V.T
        self.P=V
        compare=[]
        for i in range(len(lamda)-1):
            temp=lamda[i]/lamda[i+1]
            compare.append(temp)
        return U,S,V,compare
    def PCAdecompose(self,k):
        T=self.T[:,:k]
        P=self.P[:,:k]
        return T,P
    
class PCR():
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def confirmPCs(self):
        pca=PCA(self.X)
        U,S,V,compare=pca.SVDdecompose()
        return compare
    def model(self,PCs):
        pca=PCA(self.X)
        U,S,V,compare=pca.SVDdecompose()
        T,P=pca.PCAdecompose(PCs)
        TtT=np.dot(T.T,T)
        inv=np.linalg.inv(TtT)
        Ainv=np.dot(inv,T.T)
        A=np.dot(Ainv,self.Y)
        self.A=np.dot(P,A)
    def predict(self,Xnew):
        ans=np.dot(Xnew,self.A)
        return ans