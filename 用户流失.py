#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[39]:


plt.rcParams['font.sans-serif'] = ['SimHei']    
plt.rcParams['axes.unicode_minus'] = False  
data=pd.read_excel(r'data.xlsx')


# In[40]:


data.info()


# In[41]:


data["用户ID"] = data["用户ID"].astype("str")
data.info()


# In[42]:


data.head()


# In[43]:


data.describe()


# In[44]:


data.isnull().sum()


# In[45]:


data = data.drop_duplicates() 


# In[46]:


import numpy as np
from scipy import stats
from scipy import stats


# In[11]:


stats.chisquare(data['最近7天的登录次数'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近7天的登录时长(小时)'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近7天的充值金额'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近7天的充值次数'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近7天的最低等级'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近7天的最高等级'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近15天的登录次数'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近15天的登录时长(小时)'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近15天的充值金额'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近15天的充值次数'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近15天的最低等级'], f_exp = data['是否为流失用户'])
stats.chisquare(data['最近15天的最高等级'], f_exp = data['是否为流失用户'])


# In[12]:


#制作柱状图，看0和1分布的情况
pinlv=data['是否为流失用户'].value_counts().head(20)
pinlv
pinlv.plot.bar()


# In[13]:


sns.boxplot(x='是否为流失用户',y='最近7天的充值金额',data=data)
plt.title('最近7天的充值金额和是否为流失用户之间的箱体图')
plt.show()


# In[14]:


sns.boxplot(x='是否为流失用户',y='最近15天的充值金额',data=data)
plt.title('最近15天的充值金额和是否为流失用户之间的箱体图')
plt.show()


# In[15]:


sns.boxplot(x='是否为流失用户',y='最近15天的登录次数',data=data)
plt.title('最近15天的充值金额和是否为流失用户之间的箱体图')
plt.show()


# In[16]:


sns.boxplot(x='是否为流失用户',y='最近7天的登录次数',data=data)
plt.title('最近15天的充值金额和是否为流失用户之间的箱体图')
plt.show()


# In[17]:


sns.scatterplot(x='最近15天的充值金额',y='最近15天的登录时长(小时)',data=data)
plt.title('最近15天的充值金额和最近15天的登录时长(小时)')
plt.show()


# In[18]:


sns.scatterplot(x='最近15天的登录时长(小时)',y='最近15天的登录次数',data=data)
plt.title('最近15天的充值金额和最近15天的登录次数')
plt.show()


# In[19]:


##模型部分
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# In[20]:


x=data.iloc[:,1:-1]
x.head()


# In[21]:


y=data['是否为流失用户']
y.head()


# In[22]:


#切分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#随机森林
rf=RandomForestClassifier()
##网格搜索参数设置
param={"n_estimators":[120,200],"max_depth":[5,8]}
#网格搜索调
gc=GridSearchCV(rf,param_grid=param,cv=5)
gc.fit(x_train,y_train)

print(gc.score(x_train,y_train))
print(gc.best_params_)


# In[23]:


y_predict=gc.predict(x_test)
y_predict


# In[24]:


a=pd.DataFrame(y_predict,columns=['c'])
a.head()
b=a['c'].value_counts()
b.plot.bar()


# In[31]:


from sklearn import metrics


# In[32]:


# 模型对测试集的预测结果auc值
fpr_gc,tpr_gc,threshold_gc = metrics.roc_curve(y_test,y_predict)   
auc_gc = metrics.auc(fpr_gc,tpr_gc)                              
score_gc = metrics.accuracy_score(y_test,y_predict)                 
print([score_gc,auc_gc])  # AUC得分and准确率


# In[ ]:





# In[ ]:




