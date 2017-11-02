from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed, FileRequired
from wtforms import TextField, StringField,SubmitField,TextAreaField,IntegerField,BooleanField,\
RadioField,SelectField,SelectMultipleField,FileField,FloatField
from flask_wtf import FlaskForm as Form
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import numpy as np
import pandas as pd
tasks=[
       {
            'id':1,
            'title':u'Buy gro',
            'description':u'Milk,Cheese,Pizza,Fruit'
        },
        {
            'id':2,
            'title':u'Learn Python',
            'description':u'Need to find '     
        }]

train=[
       {
        'x_train_name':['Steam_in_Q','Steam_in_T','Steam_in_P','Steam_out_Q','P'],
        'y_train_name':u'ss'
        }]

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
global table1
class Submit(Form):
#    user_email = StringField("email address",[validators.Email()])
#    api = StringField("api",[DataRequired()])
#    submit = SubmitField("Submit")
#    code = IntegerField("code example: 200",[DataRequired()])
#    alias = StringField("alias for api")
#    data = TextAreaField("json format",[DataRequired()])
    s_in_Q=StringField('主蒸汽流量',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    s_in_T=StringField('主蒸汽温度',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    s_in_P=StringField('主蒸汽压力',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    s_out_Q=StringField('热网抽汽流量',[DataRequired()],render_kw={\
                                "style":"width:280px"})
    submit=SubmitField('提交',render_kw={"style":"class:btn btn-primary"})
    
class Get_Data_Submit_from_database(Form):
    db_name=StringField('数据库名',[DataRequired()],render_kw={\
                                "style":"width:100px"})
    table_name=StringField('表名',[DataRequired()],render_kw={\
                                "style":"width:100px"})
    submit=SubmitField('读取数据库数据',render_kw={"style":"class:btn btn-primary"})

class Get_Data_Submit_from_file(Form):
    excel_name=StringField('文件路径',[DataRequired()],render_kw={\
                                "style":"width:300px"})
    submit=SubmitField('读取文件数据',render_kw={"style":"class:btn btn-primary"})

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

class Demand_Forcast_dataload(Form):
    excel_path=StringField('EXCEL文件路径',[DataRequired()],render_kw={\
                             "style":"width:300px"})
    
    start_year=IntegerField('开始年份',[DataRequired()],render_kw={\
                               "style":"width:70px"})
    start_month=IntegerField('开始月份',[DataRequired()],render_kw={\
                             "style":"width:70px"})
    end_year=IntegerField('结束年份',[DataRequired()],render_kw={\
                               "style":"width:70px"})
    end_month=IntegerField('结束月份',[DataRequired()],render_kw={\
                       "style":"width:70px"})
    forcast_number=IntegerField('预测月数',[DataRequired()],render_kw={\
                       "style":"width:70px"})
    selected_method=SelectField('预测方法',choices=[('ARMA_METHOD','ARMA'),('DEPOSE_PLUS_METHOD','加法分解预测'),('DEPOSE_MUL_METHOD','乘法分解预测')],render_kw={\
                       "style":"width:200px"})
    read_data_buttom=SubmitField('数据提取',render_kw={"style":"class:btn btn-primary"})

class outlier_detect_dataload(Form):
    excel_path=StringField('EXCEL文件路径',[DataRequired()],render_kw={\
                             "style":"width:300px"})
    
    start_year=IntegerField('开始年份',[DataRequired()],render_kw={\
                               "style":"width:70px"})
    start_month=IntegerField('开始月份',[DataRequired()],render_kw={\
                             "style":"width:70px"})
    end_year=IntegerField('结束年份',[DataRequired()],render_kw={\
                               "style":"width:70px"})
    end_month=IntegerField('结束月份',[DataRequired()],render_kw={\
                       "style":"width:70px"})
    out_percent=FloatField('上下限百分比',[DataRequired()],render_kw={\
                       "style":"width:70px"})
    read_data_buttom=SubmitField('数据提取',render_kw={"style":"class:btn btn-primary"})


class demand_forcast_algorithms():
    """docstring for demand_forcast_algorithms"""
    #ARMA预测算法
    def arma_predict(self,data,his_start_year,his_start_month,his_end_year,his_end_month,forcast_number):
        his_start_time=str(his_start_year)+'m'+str(his_start_month)
        his_end_time=str(his_end_year)+'m'+str(his_end_month)

        temp_month=his_end_month+forcast_number
        if temp_month%12==0:
            temp_year=temp_month//12-1
            temp_month=12
        else: 
            temp_year=temp_month//12
            temp_month=temp_month%12  

        if his_end_month==12:
            forcast_start_time=str(his_end_year+1)+'m1'
        else:
            forcast_start_time=str(his_end_year)+'m'+str(his_end_month+1)
        forcast_end_time=str(his_end_year+temp_year)+'m'+str(temp_month)

        data=data.T
        data.index = pd.Index(sm.tsa.datetools.dates_from_range(his_start_time, his_end_time))
        result=pd.DataFrame(np.zeros([forcast_number,len(data.columns)]))
        regress_error=np.zeros([1,len(data.columns)])

        result.index = pd.Index(sm.tsa.datetools.dates_from_range(forcast_start_time, forcast_end_time))
        for i in range(0,len(data.columns)):
            arma_mod20 = sm.tsa.ARMA(data[i], (2,0)).fit(disp=False)
            result[i]=arma_mod20.predict(forcast_start_time, forcast_end_time, dynamic=True)
            regress_error[0,i]=np.mean(np.abs(arma_mod20.resid)/data[i])
            print(regress_error[0,i])
        regress_error=pd.DataFrame(regress_error)
        return result.T,regress_error.T
    #乘法分解预测
    def multiply_forcast_model(self,total_data,start_year,start_month,end_year,end_month,forcast_months):
        temp_month=end_month+forcast_months
        if temp_month%12==0:
            temp_year=temp_month//12-1
            temp_month=12
        else: 
            temp_year=temp_month//12
            temp_month=temp_month%12  

        if end_month==12:
            forcast_start_time=str(end_year+1)+'m1'
        else:
            forcast_start_time=str(end_year)+'m'+str(end_month+1)
        forcast_end_time=str(end_year+temp_year)+'m'+str(temp_month)

        predict_time_period=pd.Index(sm.tsa.datetools.dates_from_range(forcast_start_time, forcast_end_time))
        
        len1=total_data.shape[1]
        train_pred=np.zeros([total_data.shape[0],len1])
        pred=np.zeros([total_data.shape[0],forcast_months])
        regress_error=np.zeros([total_data.shape[0],1])
        #data=np.loadtxt('D:/Program Files/MATLAB/R2016b/bin/中控程序/需求预测模型/test.txt')
        for product_num in range(0,total_data.shape[0]):
            data=total_data[product_num,:].reshape([1,-1])
            T_C=np.zeros([1,len1])
            S_I=np.zeros([1,len1])
            for i in range(1,len1-1):
                T_C[0,i]=(data[0,i-1]+data[0,i]+data[0,i+1])/3#移动平均量
                S_I[0,i]=data[0,i]/T_C[0,i]#季节分量

            S_I_changed=np.append(np.zeros([1,start_month-1]),S_I)
            S_I_changed=np.append(S_I_changed,np.zeros([1,12-end_month]))
            temp=np.zeros([end_year-start_year+1,12])
            i=0
            for j in range(0,end_year-start_year+1):
                for k in range(0,12):
                    temp[j][k]=S_I_changed[i]
                    i+=1
            S_I_changed=temp
            del temp
        

            #计算平均S_I季节指标
            start_month_S_I=start_month+1
            end_month_S_I=end_month-1
            mean_S_I=np.zeros([1,12])
            if start_month_S_I<end_month_S_I:
                for i in range(0,12):
                    if (i+1<start_month_S_I or i+1>end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                    elif (i+1>=start_month_S_I and i+1<=end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year+1)

            if start_month_S_I==end_month_S_I:
                for i in range(0,12):
                    if (i+1<start_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                    elif ((i+1)==start_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year+1)
                    else: 
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)


            if start_month_S_I>end_month_S_I:
                for i in range(0,12):
                    if (i+1<=end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                    elif (i+1<start_month_S_I and i+1>end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year-1)
                    else: 
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                
            #计算修正的S_I平均数
            percent_S_I=12/np.sum(mean_S_I)
            fixed_mean_S_I=percent_S_I*mean_S_I
            #计算线性长期趋势系数
            index=np.linspace(1,len1,len1)
            index=index.reshape(len1,1)
            index=np.concatenate((np.ones([len1,1]),index),axis=1)
            lin_model=LinearRegression()
            lin_model.fit(index,data.T)


            T=np.zeros([1,len1])
            for i in range(0,len1):
                temp1=[[1,i+1]]
                T[0,i]=np.float(lin_model.predict(np.array(temp1)))
            #求循环分量
            C=T_C/T
            C_changed=np.append(np.zeros([1,start_month-1]),C)
            C_changed=np.append(C_changed,np.zeros([1,12-end_month]))
            temp=np.zeros([end_year-start_year+1,12])
            i=0
            for j in range(0,end_year-start_year+1):
                for k in range(0,12):
                    temp[j][k]=C_changed[i]
                    i+=1
            C_changed=temp
            del temp

            #计算平均循环分量月份指标
            start_month_C=start_month+1
            end_month_C=end_month-1
            mean_C=np.zeros([1,12])
            if start_month_C<end_month_C:
                for i in range(0,12):
                    if (i+1 <start_month_C or i+1>end_month_C):
                        mean_C[0,i]=np.sum(C_changed[:,i])/(end_year-start_year)
                    elif (i+1>=start_month_C and  i+1<=end_month_C):
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year+1)

            if start_month_C==end_month_C:
                for i in range(0,12):
                    if (i+1<start_month_C):
                        mean_C[0,i]=np.sum(C_changed[:,i])/(end_year-start_year)
                    elif ((i+1)==start_month_C):
                        mean_C[0,i]=np.sum(C_changed[:,i])/(end_year-start_year+1)
                    else: 
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year)
                
            if start_month_C>end_month_C:
                for i in range(0,12):
                    if (i+1<=end_month_C):
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year)
                    elif (i+1<start_month_C and i+1>end_month_C):
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year-1)
                    else: 
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year)

            percent_C=12/np.sum(mean_C)
            fixed_mean_C=percent_C*mean_C

            #进行预测

            for i in range(0,len1):
                current_month=start_month+i
                tem11=[[1,i+1]]
                current_T=np.float(lin_model.predict(np.array(tem11)))
                month1=np.mod(current_month,12)
                if month1==0:
                    current_C=fixed_mean_C[0,11]
                    current_S=fixed_mean_S_I[0,11]
                else:
                    current_C=fixed_mean_C[0,month1-1]
                    current_S=fixed_mean_S_I[0,month1-1]
                train_pred[product_num,i]=np.float(current_T)*np.float(current_C)*np.float(current_S)
            
            regress_error[product_num,0]=np.mean(np.abs(train_pred[product_num,:].reshape(1,-1)-data)/data)

            for i in range(0,forcast_months):
                current_month=end_month+i+1
                tem11=[[1,len1+i+1]]
                current_T=np.float(lin_model.predict(np.array(tem11)));
                month1=np.mod(current_month,12)
                if month1==0:
                    current_C=fixed_mean_C[0,11]
                    current_S=fixed_mean_S_I[0,11]
                else:
                    current_C=fixed_mean_C[0,month1-1]
                    current_S=fixed_mean_S_I[0,month1-1]
                pred[product_num,i]=np.float(current_T)*np.float(current_C)*np.float(current_S)
        pred=pd.DataFrame(pred)
        regress_error=pd.DataFrame(regress_error)
        return pred,regress_error,predict_time_period
    #加法分解预测
    def plus_forcast_model(self,total_data,start_year,start_month,end_year,end_month,forcast_months):
        temp_month=end_month+forcast_months
        if temp_month%12==0:
            temp_year=temp_month//12-1
            temp_month=12
        else: 
            temp_year=temp_month//12
            temp_month=temp_month%12  

        if end_month==12:
            forcast_start_time=str(end_year+1)+'m1'
        else:
            forcast_start_time=str(end_year)+'m'+str(end_month+1)
        forcast_end_time=str(end_year+temp_year)+'m'+str(temp_month)

        predict_time_period=pd.Index(sm.tsa.datetools.dates_from_range(forcast_start_time, forcast_end_time))
        
        len1=total_data.shape[1]
        train_pred=np.zeros([total_data.shape[0],len1])
        pred=np.zeros([total_data.shape[0],forcast_months])
        regress_error=np.zeros([total_data.shape[0],1])
        #data=np.loadtxt('D:/Program Files/MATLAB/R2016b/bin/中控程序/需求预测模型/test.txt')
        for product_num in range(0,total_data.shape[0]):
            data=total_data[product_num,:].reshape([1,-1])
            T_C=np.zeros([1,len1])
            S_I=np.zeros([1,len1])
            for i in range(1,len1-1):
                T_C[0,i]=(data[0,i-1]+data[0,i]+data[0,i+1])/3#移动平均量
                S_I[0,i]=data[0,i]/T_C[0,i]#季节分量

            S_I_changed=np.append(np.zeros([1,start_month-1]),S_I)
            S_I_changed=np.append(S_I_changed,np.zeros([1,12-end_month]))
            temp=np.zeros([end_year-start_year+1,12])
            i=0
            for j in range(0,end_year-start_year+1):
                for k in range(0,12):
                    temp[j][k]=S_I_changed[i]
                    i+=1
            S_I_changed=temp
            del temp
        

            #计算平均S_I季节指标
            start_month_S_I=start_month+1
            end_month_S_I=end_month-1
            mean_S_I=np.zeros([1,12])
            if start_month_S_I<end_month_S_I:
                for i in range(0,12):
                    if (i+1<start_month_S_I or i+1>end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                    elif (i+1>=start_month_S_I and i+1<=end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year+1)

            if start_month_S_I==end_month_S_I:
                for i in range(0,12):
                    if (i+1<start_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                    elif ((i+1)==start_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year+1)
                    else: 
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)


            if start_month_S_I>end_month_S_I:
                for i in range(0,12):
                    if (i+1<=end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                    elif (i+1<start_month_S_I and i+1>end_month_S_I):
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year-1)
                    else: 
                        mean_S_I[0,i]=np.sum(S_I_changed[:,i])/(end_year-start_year)
                
            #计算修正的S_I平均数
            percent_S_I=12/np.sum(mean_S_I)
            fixed_mean_S_I=percent_S_I*mean_S_I
            #计算线性长期趋势系数
            index=np.linspace(1,len1,len1)
            index=index.reshape(len1,1)
            index=np.concatenate((np.ones([len1,1]),index),axis=1)
            lin_model=LinearRegression()
            lin_model.fit(index,data.T)


            T=np.zeros([1,len1])
            for i in range(0,len1):
                temp1=[[1,i+1]]
                T[0,i]=np.float(lin_model.predict(np.array(temp1)))
            #求循环分量
            C=data-T-S_I
            C_changed=np.append(np.zeros([1,start_month-1]),C)
            C_changed=np.append(C_changed,np.zeros([1,12-end_month]))
            temp=np.zeros([end_year-start_year+1,12])
            i=0
            for j in range(0,end_year-start_year+1):
                for k in range(0,12):
                    temp[j][k]=C_changed[i]
                    i+=1
            C_changed=temp
            del temp

            #计算平均循环分量月份指标
            start_month_C=start_month+1
            end_month_C=end_month-1
            mean_C=np.zeros([1,12])
            if start_month_C<end_month_C:
                for i in range(0,12):
                    if (i+1 <start_month_C or i+1>end_month_C):
                        mean_C[0,i]=np.sum(C_changed[:,i])/(end_year-start_year)
                    elif (i+1>=start_month_C and  i+1<=end_month_C):
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year+1)

            if start_month_C==end_month_C:
                for i in range(0,12):
                    if (i+1<start_month_C):
                        mean_C[0,i]=np.sum(C_changed[:,i])/(end_year-start_year)
                    elif ((i+1)==start_month_C):
                        mean_C[0,i]=np.sum(C_changed[:,i])/(end_year-start_year+1)
                    else: 
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year)
                
            if start_month_C>end_month_C:
                for i in range(0,12):
                    if (i+1<=end_month_C):
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year)
                    elif (i+1<start_month_C and i+1>end_month_C):
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year-1)
                    else: 
                        mean_C[0,i]=sum(C_changed[:,i])/(end_year-start_year)

            percent_C=12/np.sum(mean_C)
            fixed_mean_C=percent_C*mean_C

            #进行预测

            for i in range(0,len1):
                current_month=start_month+i
                tem11=[[1,i+1]]
                current_T=np.float(lin_model.predict(np.array(tem11)))
                month1=np.mod(current_month,12)
                if month1==0:
                    current_C=fixed_mean_C[0,11]
                    current_S=fixed_mean_S_I[0,11]
                else:
                    current_C=fixed_mean_C[0,month1-1]
                    current_S=fixed_mean_S_I[0,month1-1]
                train_pred[product_num,i]=np.float(current_T)+np.float(current_C)+np.float(current_S)
            
            regress_error[product_num,0]=np.mean(np.abs(train_pred[product_num,:].reshape(1,-1)-data)/data)

            for i in range(0,forcast_months):
                current_month=end_month+i+1
                tem11=[[1,len1+i+1]]
                current_T=np.float(lin_model.predict(np.array(tem11)));
                month1=np.mod(current_month,12)
                if month1==0:
                    current_C=fixed_mean_C[0,11]
                    current_S=fixed_mean_S_I[0,11]
                else:
                    current_C=fixed_mean_C[0,month1-1]
                    current_S=fixed_mean_S_I[0,month1-1]
                pred[product_num,i]=np.float(current_T)+np.float(current_C)+np.float(current_S)
        pred=pd.DataFrame(pred) 
        regress_error=pd.DataFrame(regress_error)   
        return pred,regress_error,predict_time_period
