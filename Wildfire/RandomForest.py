#!/usr/bin/env python
# coding: utf-8

# # 使用随机森林预测森林火灾发生概率
# 利用机器学习对森林火灾的是否会发生做出预测，并分析的模型的错判率。
# 使用从葡萄牙东北部的Montesinho国家公园（517条）、阿尔及利亚东北部Bejaia区域（122条）和阿尔及利亚西北部Sidi Belabbes区域（122条）采集的最新数据预测森林火灾的受灾面积。应用决策树和随机森林对三类指标进行分析(即时间，气象指标和部分FWI系统指标)。将对三类不同性质的指标分别进行基于机器学习的数据分析，如气象指标(即温度，相对湿度，风速和降雨量)与随机森林相结合，能够预测森林火灾是否会发生，构建火灾燃烧等级对未来的火灾防治和消防管理决策是非常有用的。
# 
# 
# 数据源网站1：https://tianchi.aliyun.com/dataset/dataDetail?dataId=92968
# 
# 数据源网站2：https://tianchi.aliyun.com/dataset/dataDetail?dataId=103992#1

# ## 1.获取数据

# In[5]:


#引入工具包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')#


# In[6]:


#读入数据
df = pd.read_csv('C:/Users/admin/Desktop/forest_fire.csv',index_col=None)


# ## 2.数据预处理

# In[7]:


#检测缺失值并清除
df.isnull().any()


# In[9]:


#数据样例
df.head()


# ## 3.分析数据

# ### 3.1 数据描述

# * 761条数据，每条数据10个特征
# * 总森火发生率54.008%，占比大致一半
# * 通过分组平均统计感知，降雨量（RF）与是否着火关系较大

# In[10]:


#数据体量与类型
df.shape


# In[11]:


df.dtypes


# In[12]:


fire_rate = df.fire.value_counts()/len(df)
fire_rate


# In[13]:


#显示统计数据
df.describe()


# In[14]:


#分组进行平均数据统计
fire_summary = df.groupby('fire')
fire_summary.mean()


# ### 3.2 相关性分析

# In[15]:


#相关性矩阵与热力图
corr = df.corr()
sns.heatmap(corr,
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values)
corr


# In[16]:


#比较着火/未着火情况下细小可燃物湿度码(FFMC)和初始蔓延速率(ISI),进行T-Test
fire_ffmc = df['FFMC'][df['fire']==1].mean()
nofire_ffmc = df['FFMC'][df['fire']==0].mean()
fire_isi = df['ISI'][df['fire']==1].mean()
nofire_isi = df['ISI'][df['fire']==0].mean()
import scipy.stats as stats
stats.ttest_1samp(a = df[df['fire']==1]['FFMC'], popmean = nofire_ffmc)


# In[17]:


stats.ttest_1samp(a = df[df['fire']==0]['ISI'],popmean = fire_isi)


# T-Test显示在是否着火的两组数据中，FFMC和ISI都是有显著差异的。
# 
# 同时观察平均值时发现降雨量(RF)的均值差距很大，而这一指标与是否着火的相关系数却并不算高（-0.11），考虑进一步使用概率密度分布图分类观察影响着火的因素，认识其数据在着/不着火情况下的分布特征。

# In[18]:


fig = plt.figure(figsize = (15,4),)
ax = sns.kdeplot(df.loc[(df['fire']==0),'RF'],color = 'g',shade = True,label = 'no fire')
ax = sns.kdeplot(df.loc[(df['fire']==1),'RF'],color = 'r',shade = True,label = 'fire')
ax.set(xlabel = 'RainFall(cm)',ylabel = 'Frequency')
plt.title('RainFall Distribution - if fire')


# In[19]:


fig = plt.figure(figsize = (15,4),)
ax = sns.kdeplot(df.loc[(df['fire']==0),'FFMC'],color = 'g',shade = True,label = 'no fire')
ax = sns.kdeplot(df.loc[(df['fire']==1),'FFMC'],color = 'r',shade = True,label = 'fire')
ax.set(xlabel = 'fine fuel moisture code(FFMC)',ylabel = 'Frequency')
plt.title('FFMC Distribution - if fire')


# In[27]:


fig = plt.figure(figsize = (15,4),)
ax = sns.kdeplot(df.loc[(df['fire']==0),'ISI'],color = 'g',shade = True,label = 'no fire')
ax = sns.kdeplot(df.loc[(df['fire']==1),'ISI'],color = 'r',shade = True,label = 'fire')
ax.set(xlabel = 'Initial Spread Index (ISI) ',ylabel = 'Frequency')
plt.title('ISI Distribution - if fire')


# In[28]:


fig = plt.figure(figsize = (15,4),)
ax = sns.kdeplot(df.loc[(df['fire']==0),'temp'],color = 'g',shade = True,label = 'no fire')
ax = sns.kdeplot(df.loc[(df['fire']==1),'temp'],color = 'r',shade = True,label = 'fire')
ax.set(xlabel = 'Temperature(℃) ',ylabel = 'Frequency')
plt.title('Temperature Distribution - if fire')


# ## 4.决策树（DecisionTree）和随机森林（RandomForest）预测模型的构建

# In[29]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,precision_score,recall_score


# In[38]:


#产生x，y；将数据分为训练和测试数据集;stratify = y 意味着在产生训练和测试数据中，
#着火案例的百分比等于原来总的数据中的着火案例的百分比。
target_name = 'fire'
x = df.drop('fire',axis=1)
y = df[target_name]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=123,stratify = y)


# In[39]:


#采用决策树和随机森林两种建模方法，使用决策树的原因是决策树模型可以可视化
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#决策树
dtree = tree.DecisionTreeClassifier(criterion = 'entropy',min_weight_fraction_leaf = 0.01) #叶子节点最少要含1%的样本
dtree = dtree.fit(x_train,y_train)
print('\n\n < 决策树 >')
dt_roc_auc = roc_auc_score(y_test,dtree.predict(x_test))
print('决策树 AUC = %2.2f' % dt_roc_auc)   
print(classification_report(y_test,dtree.predict(x_test)))
                                    
#随机森林                                 
rf = RandomForestClassifier(
criterion = 'entropy',#使用信息熵计算信息增益，进行分类,也可以采用基尼不纯度'gini'
n_estimators = 1000,#选择构建树的数量
max_depth = None,#不使用设置深度的方式来防止过拟合，因为不明确拟合情况
min_samples_split = 10,#定义至少有10个样本才继续分叉
#min_weight_fraction_leaf = 0.02 #也可以采用定义叶子结点最少需要包含多少百分比的样本来防止过拟合
)
rf.fit(x_train,y_train)
print('\n\n < 随机森林 >')
rf_roc_auc = roc_auc_score(y_test,rf.predict(x_test))
print('随机森林 AUC = %2.2f' % rf_roc_auc)
print(classification_report(y_test,rf.predict(x_test)))


# # 5.模型对比与特征选取

# ### 5.1 ROC图

# In[40]:


from sklearn.metrics import roc_curve
rf_fpr,rf_tpr,rf_thresholds = roc_curve(y_test,rf.predict_proba(x_test)[:,1])
dt_fpr,dt_tpr,dt_thresholds = roc_curve(y_test,dtree.predict_proba(x_test)[:,1])

plt.plot(rf_fpr,rf_tpr,label = 'RadomForest(area = %0.2f)'%rf_roc_auc)
plt.plot(dt_fpr,dt_tpr,label = 'DecisionTree(area = %0.2f)'%dt_roc_auc)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc = 'lower right')
plt.show


# ### 5.2 选择随机森林辅助特征选择

# In[43]:


importances = rf.feature_importances_
feat_names = df.drop(['fire'],axis = 1).columns

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title = ('Feature importances by Random Forest')
plt.bar(range(len(indices)),importances[indices],color='darkgreen',align = 'center')
plt.step(range(len(indices)),np.cumsum(importances[indices]),color='green',where = 'mid',label = 'Cumulative')
plt.xticks(range(len(indices)),feat_names[indices],rotation = 'vertical',fontsize = 14)
plt.xlim([-1,len(indices)])
plt.show

