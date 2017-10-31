from flask import render_template,redirect,request,url_for,flash,session
from . import main
from flask_wtf import FlaskForm as Form
from .forms import NameForm,Get_Data_Submit,Samping_Submit,Missing_Process_Submit,Describe_Statistics_Submit,\
Pearson_cal_Submit,Hist_Graph,Selected_1,Selected_2,MultipleSelect,NeuralNet,RandomForest
from ..models import Get_Data,Statistics,PCA,PCR
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
import xlrd
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn import preprocessing

import random
from pyecharts import Scatter3D
from pyecharts.constants import DEFAULT_HOST
from pyecharts import Bar
import scipy.stats as sts

data_num=0
data=[]
samples=[]
hist_selected_column=""
Selected_1_default_item=0
Selected_2_default_item=0
x=[]
y=[]
x_linear_regression=[]
y_linear_regression=[]
choices=[]

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/preprocess',methods=['GET','POST'])
def preprocess():
    global data_num,data,samples
    form1=Get_Data_Submit()
    if form1.validate_on_submit():
        db_name=form1.db_name.data
        table_name=form1.table_name.data
        GD=Get_Data()
        columns,data_count,data=GD.get_data(db_name,table_name)
        for i in range(len(columns)):
            session[('columns_name'+str(i))]=columns[i][0]
            session[('columns_type'+str(i))]=columns[i][1]
            session[('data_count'+str(i))]=data_count[0][0]

        
        data_num=data_count[0][0]
    form2=Samping_Submit()
    if form2.validate_on_submit(): 
        if form2.sampling_ratio.data!="":
            sampling_ratio=float(form2.sampling_ratio.data)
            form2.sampling_count.data=int(sampling_ratio*data_num)
        if form2.sampling_count.data!="":
            sampling_ratio=int(form2.sampling_count.data)/data_num
            form2.sampling_ratio.data=str(sampling_ratio)
            s_k=random.sample(range(data_num),int(form2.sampling_count.data))
            
            samples=data.iloc[s_k,:]

        
    form3=Missing_Process_Submit()
#    if form3.validate_on_submit():
#        form4=Hist_Graph()
#        return render_template('statistics.html',form2=form4)

    return render_template('preprocess.html',form1=form1, form2=form2,form3=form3,\
                           columns_name0=session.get('columns_name0'),columns_name1=session.get('columns_name1'),\
                           columns_name2=session.get('columns_name2'),columns_name3=session.get('columns_name3'),\
                           columns_name4=session.get('columns_name4'),columns_name5=session.get('columns_name5'),\
                           columns_name6=session.get('columns_name6'),columns_type0=session.get('columns_type0'),\
                           columns_type1=session.get('columns_type1'),\
                           columns_type2=session.get('columns_type2'),columns_type3=session.get('columns_type3'),\
                           columns_type4=session.get('columns_type4'),columns_type5=session.get('columns_type5'),\
                           columns_type6=session.get('columns_type6'),\
                           data_count0=session.get('data_count0'),data_count1=session.get('data_count1'),\
                           data_count2=session.get('data_count2'),data_count3=session.get('data_count3'),\
                           data_count4=session.get('data_count4'),data_count5=session.get('data_count5'),\
                           data_count6=session.get('data_count6'))

@main.route('/statistics',methods=['GET','POST']) 
def statistics():
    global choices
    global Selected_2_default_item
    global x
    global y
    form2=Hist_Graph()
    choices=list(zip(range(len(samples.columns)),samples.columns))
    form2.item.choices=choices

    if form2.submit1.data and form2.validate_on_submit():
        session['select_item']=str(samples.columns[form2.item.data])
        global hist_selected_column
        hist_selected_column=str(samples.columns[form2.item.data])
#        return render_template('statistics.html',form2=form2,select_item=session.get('select_item'))

    form3=Selected_1()
    form3.item2.choices=choices    
    
    form4=Selected_2()
    form4.item3.choices=choices 
    
    if form3.submit2.data and form3.validate_on_submit():
#        Selected_2_default_item=3
        session['Selected_X']=str(samples.columns[form3.item2.data])
        x=samples[str(samples.columns[form3.item2.data])]
#        print(x)
        
    if form4.submit3.data and form4.validate_on_submit():
#        Selected_1_default_item=4
        session['Selected_Y']=str(samples.columns[form4.item3.data])
        y=samples[str(samples.columns[form4.item3.data])]
    
    S=Statistics()
    
    form_pearson=Pearson_cal_Submit()
    if form_pearson.submit_pearson.data and form_pearson.validate_on_submit():
        spearman_R,spearman_p=S.spearman(x,y)
        pearson_R,pearson_p=S.pearson(x,y)
        session['pearson']=str(pearson_R)
    
    form1=Describe_Statistics_Submit()
    if form1.submit.data and form1.validate_on_submit():
        count,mean,std,mn,mx,per25,per50,per75,cv,kurtosis,skewness=S.descriptive_statistics(samples)
        for i in range(len(samples.columns)):
            session[('columns'+str(i))]=samples.columns[i]
            session['s1_'+str(i)]=count[i]
            session['s2_'+str(i)]=mean[i]
            session['s3_'+str(i)]=std[i]
            session['s4_'+str(i)]=mn[i]
            session['s5_'+str(i)]=mx[i]
            session['s6_'+str(i)]=per25[i]
            session['s7_'+str(i)]=per50[i]
            session['s8_'+str(i)]=per75[i]
            session['s9_'+str(i)]=cv[i]
            session['s10_'+str(i)]=kurtosis[i]
            session['s11_'+str(i)]=skewness[i]
     
    return render_template('statistics.html',form1=form1,form2=form2,form3=form3,form4=form4,form_pearson=form_pearson,\
                           columns_name0=session.get('columns_name0'),columns_name1=session.get('columns_name1'),\
                           columns_name2=session.get('columns_name2'),columns_name3=session.get('columns_name3'),\
                           columns_name4=session.get('columns_name4'),\
                           s1_0=session.get('s1_0'),s1_1=session.get('s1_1'),s1_2=session.get('s1_2'),s1_3=session.get('s1_3'),s1_4=session.get('s1_4'),\
                           s2_0=session.get('s2_0'),s2_1=session.get('s2_1'),s2_2=session.get('s2_2'),s2_3=session.get('s2_3'),s2_4=session.get('s2_4'),\
                           s3_0=session.get('s3_0'),s3_1=session.get('s3_1'),s3_2=session.get('s3_2'),s3_3=session.get('s3_3'),s3_4=session.get('s3_4'),\
                           s4_0=session.get('s4_0'),s4_1=session.get('s4_1'),s4_2=session.get('s4_2'),s4_3=session.get('s4_3'),s4_4=session.get('s4_4'),\
                           s5_0=session.get('s5_0'),s5_1=session.get('s5_1'),s5_2=session.get('s5_2'),s5_3=session.get('s5_3'),s5_4=session.get('s5_4'),\
                           s6_0=session.get('s6_0'),s6_1=session.get('s6_1'),s6_2=session.get('s6_2'),s6_3=session.get('s6_3'),s6_4=session.get('s6_4'),\
                           s7_0=session.get('s7_0'),s7_1=session.get('s7_1'),s7_2=session.get('s7_2'),s7_3=session.get('s7_3'),s7_4=session.get('s7_4'),\
                           s8_0=session.get('s8_0'),s8_1=session.get('s8_1'),s8_2=session.get('s8_2'),s8_3=session.get('s8_3'),s8_4=session.get('s8_4'),\
                           s9_0=session.get('s9_0'),s9_1=session.get('s9_1'),s9_2=session.get('s9_2'),s9_3=session.get('s9_3'),s9_4=session.get('s9_4'),\
                           s10_0=session.get('s10_0'),s10_1=session.get('s10_1'),s10_2=session.get('s10_2'),s10_3=session.get('s10_3'),s10_4=session.get('s10_4'),\
                           s11_0=session.get('s11_0'),s11_1=session.get('s11_1'),s11_2=session.get('s11_2'),s11_3=session.get('s11_3'),s11_4=session.get('s11_4'),\
                           select_item=session.get('select_item'),Selected_X=session.get('Selected_X'),Selected_Y=session.get('Selected_Y'),\
                           pearson=session.get('pearson'))
    
@main.route('/feature')
def feature():
    return render_template('feature.html')

@main.route('/regression',methods=['GET','POST'])
def regression():
    global choices
    global x_linear_regression
    global y_linear_regression
    form3=Selected_1()
#    form3.item2.description='因变量ww'
    form3.item2.choices=choices    
    
    form_multipleSelect=MultipleSelect()
    form_multipleSelect.item_multiple.choices=choices
    
    if form3.submit2.data and form3.validate_on_submit():
        session['single_selected']=str(samples.columns[form3.item2.data])
        y_linear_regression=samples[str(samples.columns[form3.item2.data])]
        line_reg=linear_model.LinearRegression()
        line_reg.fit(x_linear_regression,y_linear_regression)
        a,b,r=line_reg.coef_,line_reg.intercept_,line_reg._residues
        R=1-r/sum((y_linear_regression-y_linear_regression.mean())**2)
        session['a']=str(a)
        session['b']=str('%.3f' %b)
        session['r']=str(r)
        session['R']=str(R)
#        print(a)
#        print(b)

    if form_multipleSelect.submit_multiple.data and form_multipleSelect.validate_on_submit():
        session['multiple_selected']=str(list(samples.columns[form_multipleSelect.item_multiple.data]))
        x_linear_regression=samples[samples.columns[form_multipleSelect.item_multiple.data]]
    
    
    form_neural=NeuralNet()
    if form_neural.submit_Neural.data and form_neural.validate_on_submit():
        start_time=time.clock()
        X_train, X_test, y_train, y_test = train_test_split(x_linear_regression, \
                                                            y_linear_regression, test_size=0.33, random_state=42)

#        Input=x_linear_regression
#        Target=y_linear_regression.values.reshape(-1,1)
        
        Input=X_train
        Target=y_train.values.reshape(-1,1)
        
        input_train_minmax_scaler=preprocessing.MinMaxScaler().fit(Input)
        input_train_scaler_transform=input_train_minmax_scaler.transform(Input)
        
        output_train_minmax_scaler=preprocessing.MinMaxScaler().fit(Target)
        output_train_scaler_transform=output_train_minmax_scaler.transform(Target)
        
        minmax=[]
        for _ in range(len(Input.columns)):
            minmax.append([0,1])
#        minmax=list(zip(Input.min(),Input.max()))
#        print(minmax)
#        print(Target)
        net_layer=[3,1]
        net=nl.net.newff(minmax,net_layer)
        train=net.train(input_train_scaler_transform,output_train_scaler_transform,epochs=500,show=100,goal=0.01)
        out=output_train_minmax_scaler.inverse_transform(net.sim(input_train_scaler_transform))
#        SSEf=error.SSE()
#        SSE=SSEf(Target,out)
        SSE=sum((Target-out)**2)
        R_neuro=1-(SSE/sum((Target-Target.mean())**2))
        end_time=time.clock()
        time_span=str("%.3f" %(end_time-start_time))
        session['time_span']=time_span
        session['SSE']=str('%.3f' %SSE)
        session['R_neuro']=str('%.4f' %R_neuro)

        
    form_RF=RandomForest()
    if form_RF.submit_RF.data and form_RF.validate_on_submit():
        start_time=time.clock()
        X_train, X_test, y_train, y_test = train_test_split(x_linear_regression, \
                                                            y_linear_regression, test_size=0.33, random_state=42)
        rf=RandomForestRegressor()
        GBDT=GradientBoostingRegressor()
        rf.fit(X_train,y_train)
        GBDT.fit(X_train,y_train)
#        y_predict=rf.predict(x_predict)
        R_RF=rf.score(X_train,y_train)
        R_GBDT=GBDT.score(X_train,y_train)
        end_time=time.clock()
        time_span_RF=str("%.3f" %(end_time-start_time))
        session['R_RF']=str('%.4f' %R_RF)
        session['time_span_RF']=time_span_RF
        session['R_GBDT']=str('%.4f' %R_GBDT)

    
    return render_template('regression.html',form3=form3,form_multipleSelect=form_multipleSelect,\
                           form_neural=form_neural,form_RF=form_RF,\
                           multiple_selected=session.get('multiple_selected'),\
                           single_selected=session.get('single_selected'),\
                           a=session.get('a'),b=session.get('b'),r=session.get('r'),R=session.get('R'),\
                           time_span=session.get('time_span'),SSE=session.get('SSE'),R_neuro=session.get('R_neuro'),\
                           R_RF=session.get('R_RF'),time_span_RF=session.get('time_span_RF'),\
                           R_GBDT=session.get('R_GBDT'))


@main.route('/fault_detection',methods=['GET','POST'])
def fault_detection():
	global fault_data,X_data,Y_data,Ypre
	X_data=[]
	Y_data=[]
	Ypre=[]
	fault_form=Get_Data_Submit()
	if fault_form.validate_on_submit():
		db_name=fault_form.db_name.data
		table_name=fault_form.table_name.data
		GD=Get_Data()
		fault_columns,fault_data_count,fault_data=GD.get_data(db_name,table_name)
		xt=pd.DataFrame(fault_data)
		Y=xt.iloc[:400,0]
		X=xt.iloc[:400,1:5]
		pcr=PCR(X,Y)
		k=3
		pcr.model(k)
		Ypre=pcr.predict(X).tolist()
		X_data=list(range(len(Y)+1))[1:]
		Y_data=Y.tolist()
	return render_template('fault_detection.html',form=fault_form,X_data=X_data,Y_data=Y_data,predict_data=Ypre)
