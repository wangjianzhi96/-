#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import random
import datetime
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from liushi_functions import *
import os

import warnings
warnings.filterwarnings("ignore")


# In[4]:


# 读入数据
creditData = pd.read_csv(file_path+'data.csv',header=0)

# 数据切分
testData, trainData = train_test_split(creditData, train_size=0.3, random_state=1)
# trainData = creditData
# 保存数据
trainData.to_csv(folderOfData+'trainData.csv',index=False)
testData.to_csv(folderOfData+'testData.csv',index=False)

# 读取切分后的数据
trainData = pd.read_csv(folderOfData+'trainData.csv',header = 0)
testData = pd.read_csv(folderOfData+'testData.csv',header = 0)


# In[5]:


print(trainData.shape)
trainData.head()


# 
#  分箱，计算WOE并编码   
# 

# In[6]:


numericalFeatures = trainData.columns.tolist()
numericalFeatures.remove('user_id')
numericalFeatures.remove('label')


# In[7]:


removed_features_1 = []
short_features = []
long_features = []

WOE_IV_dict = {}

for fea in numericalFeatures:
    if len(set(trainData[fea])) == 1:
        removed_features_1.append(fea)
    elif len(set(trainData[fea])) <=5:
        short_features.append(fea)
    else:
        long_features.append(fea)

for fea in removed_features_1:
    numericalFeatures.remove(fea)


# In[8]:



var_cutoff = {}
unchanged_features = []
notmon_features = []
for col in short_features:
    BRM = BadRateMonotone(trainData, col, 'label')
    if BRM:
        unchanged_features.append(col)
    else:
        notmon_features.append(col)
        new_col = col + '_Bin'
        trainData[new_col] = trainData[col].apply(lambda x: int(x >= 0))


# In[9]:


numericalFeatures


# In[10]:


bin_dict = []
deleted_features = []   
bin_features = []

for col in numericalFeatures:

    print("{} is in processing".format(col))
    col1 = str(col) + '_Bin'

    cutOffPoints = ChiMerge(trainData, col, 'label')
    var_cutoff[col] = cutOffPoints
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints))
    bin_dict.append({col:cutOffPoints})

    bin_merged = Monotone_Merge(trainData, 'label', col1)
    removed_index = []
    for bin in bin_merged:
        if len(bin)>1:
            indices = [int(b.replace('Bin ','')) for b in bin]
            removed_index = removed_index+indices[0:-1]
    removed_point = [cutOffPoints[k] for k in removed_index]
    for p in removed_point:
        cutOffPoints.remove(p)

    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints))
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        del trainData[col1]
        print('we delete {} because the maximum bin occupies more than 90%'.format(col))
    else:
        var_cutoff[col] = cutOffPoints
        bin_features.append(col1)


# In[11]:


var_IV = {}  # save the IV values for binned features   
var_WOE = {}
#remove_features = ['order_status_5_7D_Bin']
# for col in bin_features+unchanged_features if col not in remove_features:
for col in bin_features:
# for col in bin_features+unchanged_features:
    WOE_IV = CalcWOE(trainData, col, 'label')
    WOE_IV_dict[col] = CalcWOE(trainData, col, 'label')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']


# In[12]:


#  选取IV高于0.02的变量
high_IV = [(k,v) for k,v in var_IV.items() if v >= 0.00]
high_IV_sorted = sorted(high_IV, key=lambda k: k[1],reverse=True)
high_IV_features = [i[0] for i in high_IV_sorted]
high_IV_values = [i[1] for i in high_IV_sorted]
for var in high_IV_features:
    newVar = var+"_WOE"
    trainData[newVar] = trainData[var].map(lambda x: var_WOE[var][x])


# In[50]:


high_IV


# In[13]:


'''
对于不需要合并、原始箱的bad rate单调的特征，直接计算WOE和IV
'''
#for var in large_bin_var:
#    WOE_IV_dict[var] = CalcWOE(trainData, var, 'label')


# In[14]:


#var_IV = {}  # save the IV values for binned features       #将IV值保留和WOE值
#var_WOE = {}
#for col in bin_features+unchanged_features:
#    WOE_IV = CalcWOE(trainData, col, 'label')
#    WOE_IV_dict[col] = CalcWOE(trainData, col, 'label')
#    var_IV[col] = WOE_IV['IV']
#    var_WOE[col] = WOE_IV['WOE']


# In[15]:


plt.bar(x=range(len(high_IV_values)), height = high_IV_values)


# 
# 单变量分析和多变量分析   

# In[16]:


'''
单变量分析：比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
'''

removed_var  = []
roh_thresould = 0.6
for i in range(len(high_IV_features)-1):
    if high_IV_features[i] not in removed_var:
        x1 = high_IV_features[i]+"_WOE"
        for j in range(i+1,len(high_IV_features)):
            if high_IV_features[j] not in removed_var:
                x2 = high_IV_features[j] + "_WOE"
                roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if var_IV[high_IV_features[i]] > var_IV[high_IV_features[j]]:
                        removed_var.append(high_IV_features[j])
                    else:
                        removed_var.append(high_IV_features[i])

multivariates = [i+"_WOE" for i in high_IV_features if i not in removed_var]


# In[17]:


dfData = trainData[multivariates].corr()
plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")


# In[18]:


X = trainData[multivariates]
X.isnull().any(axis=0)


# In[19]:


trainData_m = trainData[multivariates]
trainData_m[np.isinf(trainData_m)] = 0


# In[20]:


X = np.mat(trainData_m)
vif_list = []
for i in range(len(multivariates)):
    vif = variance_inflation_factor(X, i)
    vif_list.append(vif)
    if vif > 10:
        print("Warning: the vif for {0} is {1}".format(high_IV_features[i], vif))
        #print("Warning: the vif for  is 1")

plt.bar(x=range(len(vif_list)), height = sorted(vif_list,reverse=True))
'''
这一步没有发现有多重共线性
'''


# In[21]:


# 去掉多重共线性变量
#multivariates.remove('huabei_overdue_history_WOE')
#multivariates.remove('jiebei_consume_quota_WOE')


# In[22]:


multivariates


# 
# 建立逻辑回归模型预测   
# 

# In[23]:


X = trainData[multivariates]
X[np.isinf(X)] = 0
X['intercept'] = [1] * X.shape[0]
y = trainData['label']
logit = sm.Logit(y, X)
logit_result = logit.fit()
pvalues = logit_result.pvalues
params = logit_result.params
fit_result = pd.concat([params,pvalues],axis=1)
fit_result.columns = ['coef','p-value']


# In[24]:


fit_result


# In[25]:


fit_result['coef'][6:7]


# In[27]:


sm.Logit(y, trainData['number_recharge_15days_Bin_WOE']).fit().params  # -0.988119


# In[28]:


multivariates.remove('number_recharge_15days_Bin_WOE')


# In[29]:


# fea = 'lowest_level_7days_Bin_WOE'
# for fea in multivariates:
#     multivariates.remove(fea)


# In[30]:


multivariates


# # 方法一：向前筛选法

# In[31]:



selected_var = [multivariates[0]]
for var in multivariates[1:]:
    try_vars = selected_var+[var]
    X_temp = trainData[try_vars].copy()
    X_temp[np.isinf(X_temp)] = 0
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    pvals, params = LR.pvalues, LR.params
    del params['intercept']
    if max(pvals)<0.1 and max(params)<0:
        selected_var.append(var)


# In[32]:


selected_var


# In[33]:


X_final = trainData[selected_var]
X_final[np.isinf(X_final)] = 0
X_final['intercept'] = [1] * X_final.shape[0]


# In[34]:


list(X_final.columns)


# In[37]:


### 计算KS值
def KS(df, score, target, plot = True):
  
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all.reset_index(level=0, inplace=True)
    #all[score] = all.index
    all = all.sort_values(by=score)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS_list = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS = max(KS_list)
    if plot:
        plt.plot(all[score], all['badCumRate'])
        plt.plot(all[score], all['goodCumRate'])
        plt.title('KS ={}%'.format(int(KS*100)))
    return KS


# In[38]:


# KS(pred_result, 'scores', 'label',plot=False)  #0.600
# roc_auc_score(trainData['label'], y_pred)   #0.823
# ROC_AUC(pred_result, 'scores', 'label')


# In[39]:


y_pred = LR.predict(X_final)

scores = Prob2Score(y_pred, 800, 10)
pred_result = pd.DataFrame({'label':trainData['label'], 'scores':scores})


# In[40]:


plt.hist(pred_result['scores'])


# In[41]:


### 计算KS值
def KS(df, score, target, plot = True):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
#    all[score] = all.index
    all.reset_index(level=0, inplace=True)
    all = all.sort_values(by=score)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS_list = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS = max(KS_list)
    if plot:
        plt.plot(all[score], all['badCumRate'])
        plt.plot(all[score], all['goodCumRate'])
        plt.title('KS ={}%'.format(int(KS*100)))
    return KS


# In[42]:


KS(pred_result, 'scores', 'label',plot=True)  #0.600


# In[43]:


roc_auc_score(trainData['label'], y_pred)


# 
# 在测试集上测试逻辑回归的结果
# 

# In[44]:


categoricalFeatures = long_features
numericalFeatures = numericalFeatures


# In[45]:


var_cutoff = {}
unchanged_features = []
notmon_features = []
for col in short_features:
    BRM = BadRateMonotone(trainData, col, 'label')
   
    if BRM:
        unchanged_features.append(col)
    else:
        notmon_features.append(col)
        new_col = col + '_Bin'
        trainData[new_col] = trainData[col].apply(lambda x: int(x >= 0))


# In[46]:


categoricalFeatures = long_features
numericalFeatures = numericalFeatures

modelFeatures = fit_result.index.tolist()
modelFeatures.remove('intercept')
modelFeatures = [i.replace('_Bin','').replace('_WOE','') for i in modelFeatures]

numFeatures = [i for i in modelFeatures if i in numericalFeatures]
charFeatures = [i for i in modelFeatures if i in categoricalFeatures]


# In[47]:


for var in modelFeatures:
    if var not in ['number_recharge_15days']:
        newBin = var+"_Bin"
        bin = [list(i.values()) for i in bin_dict if var in i][0][0]
        testData[newBin] = testData[var].apply(lambda x: AssignBin(x, bin))


# In[48]:


testData.columns


# In[51]:


finalFeatures = []

for var in modelFeatures:
    if var not in ['number_recharge_15days']:
        var1 = var + '_Bin'
        var2 = var1+"_WOE"
        finalFeatures.append(var2)
        testData[var2] = testData[var1].apply(lambda x: WOE_IV_dict[var1]['WOE'][x])
print(finalFeatures)


# In[52]:


def Predict_LR(x, var_list, coef_dict, prob=False):
  
    s = coef_dict['intercept']
    for var in var_list:
        s += x[var]*coef_dict[var]
    if prob == True:
        y = 1.0/(1+np.exp(-s))
        return y
    else:
        return s


# In[53]:


coef_dict = params.to_dict()
coef_dict['intercept'] = 0.856158
coef_dict


# In[54]:


testData['log_odds'] = testData.apply(lambda x: Predict_LR(x, finalFeatures, coef_dict),axis=1)

perf_model = KS_AR(testData, 'log_odds', 'label')

# KS＝64.49%， AR ＝ 68.64%，都高于30%的标准。因此该模型是可用的。


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




