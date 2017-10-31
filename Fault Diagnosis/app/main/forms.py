from flask_wtf import Form
from wtforms import TextField, StringField,SubmitField,TextAreaField,IntegerField,BooleanField,\
RadioField,SelectField,SelectMultipleField
from wtforms.validators import Required
from wtforms import fields
from wtforms.validators import DataRequired
from wtforms import validators
from flask_wtf import FlaskForm as Form
#from flask_wtf import Form
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

class NameForm(Form):
    name=StringField('What is your name?',validators=[Required()])
    submit=SubmitField('Submit')

class Get_Data_Submit(Form):
    db_name=StringField('数据库名',[DataRequired()],render_kw={\
                                "style":"width:100px"})
    table_name=StringField('表名',[DataRequired()],render_kw={\
                                "style":"width:100px"})
    submit=SubmitField('读取数据',render_kw={"style":"class:btn btn-primary"})
    
class Samping_Submit(Form):
    sampling_count=StringField('采样个数')
    sampling_ratio=StringField('采样比例')
    Repetition=BooleanField('放回采样')
    submit=SubmitField('采样',render_kw={"style":"class:btn btn-primary"})
    
    
class Missing_Process_Submit(Form):
    Missing_Process=RadioField('缺失值处理',choices=[(1,'删除缺失值'),(2,'删除包含缺失值的行'),(3,'删除所有字段都是缺失值的行'),\
                                                (4,'删除只针对xx字段包含缺失值的行')],default=1)
    submit=SubmitField('处理',render_kw={"style":"class:btn btn-primary"})

class Describe_Statistics_Submit(Form):
    submit=SubmitField('处理',render_kw={"style":"class:btn btn-primary"})
    
class Pearson_cal_Submit(Form):
    submit_pearson=SubmitField('处理',render_kw={"style":"class:btn btn-primary"})
class Hist_Graph(Form):
    item=SelectField("选择列表",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=0)
    submit1=SubmitField("提交")    

class Selected_1(Form):
    global Selected_1_default_item
    item2=SelectField("因变量",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=Selected_1_default_item)
    submit2=SubmitField("提交")

class Selected_2(Form):
    global Selected_2_default_item
    item3=SelectField("自变量",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=Selected_2_default_item)
    submit3=SubmitField("提交")

class MultipleSelect(Form):
    item_multiple=SelectMultipleField("自变量",coerce=int,choices=[(0,'1'),(1,'2'),(2,'3'),(3,'4'),(4,'5')],default=0)
    submit_multiple=SubmitField('提交')    
    
class NeuralNet(Form):
    layers_num=StringField('网络层数')
    neuron_num=StringField('隐层神经元数')
    error_goal=StringField('期望误差')
    epochs=StringField('最大迭代次数')
    submit_Neural=SubmitField('计算',render_kw={"style":"class:btn btn-primary"})

class RandomForest(Form):
    n_estimators=StringField('树的数目')
    max_features=StringField('最大选取特征数')
    max_depth=StringField('树的最大深度')
    min_samples_split=StringField('节点最小分裂数')
    min_samples_leaf=StringField('叶子节点上的最小样本数')
    submit_RF=SubmitField('计算',render_kw={"style":"class:btn btn-primary"})

