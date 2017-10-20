from wtforms.validators import DataRequired
from wtforms import TextField, StringField,SubmitField,TextAreaField,IntegerField,BooleanField,\
RadioField,SelectField,SelectMultipleField
from flask_wtf import FlaskForm as Form
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
excel_data=[]
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
                                "style":"width:50px"})
    start_month=IntegerField('开始月份',[DataRequired()],render_kw={\
                                "style":"width:50px"})
    end_year=IntegerField('结束年份',[DataRequired()],render_kw={\
                                "style":"width:50px"})
    end_month=IntegerField('结束月份',[DataRequired()],render_kw={\
                                "style":"width:50px"})
    read_data_buttom=SubmitField('数据提取',render_kw={"style":"class:btn btn-primary"})



