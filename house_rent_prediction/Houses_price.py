#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[3]:


df = pd.read_csv(r"C:\Users\Usuario\Desktop\Projects\DataAnalysis\Datasets\House_Rent_Dataset.csv")


# In[4]:


df.size


# In[5]:


df.head(15)


# In[6]:


df.describe()


# In[7]:


df_col = df[["BHK","Size","Rent","Floor","City","Furnishing Status","Tenant Preferred","Bathroom","Point of Contact"]]
df_price = df[["Rent"]]


# In[8]:


for i in df_col.columns: #numero de cases amb la característica del títol
    plt.hist(df_col[i])
    plt.title(i)
    plt.show()


# In[9]:


for i in df_col.columns: #numero de cases amb la característica del títol
    plt.scatter(df_col[i],df_price)
    plt.title(i)
    plt.show()


# In[10]:


plt.figure(figsize = (16, 6))
plt.ticklabel_format(style = 'plain')
plt.scatter(df["Size"],df["Rent"])
plt.xlabel("Size")
plt.ylabel("Rent");


# In[11]:


print(df_col.corr())
sns.heatmap(df_col.corr())


# In[12]:


#Grafic de barres ciutat/preu 

plt.figure(figsize = (20, 7))
sns.barplot(x = df["City"], y = df["Rent"], palette = "nipy_spectral");


# In[13]:


#recompte de nombre de cases amb un BHK específic

sns.set_context("poster", font_scale = .8)
plt.figure(figsize = (30, 10))
ax = df["BHK"].value_counts().plot(kind = 'bar', color = "Blue", rot = 0)

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'Black')


# In[14]:


#recompte de nombre de cases amb un nº de lavavos específic

sns.set_context("poster", font_scale = .8)
plt.figure(figsize = (30, 10))
ax = df["Bathroom"].value_counts().plot(kind = 'bar', color = "Green", rot = 0)

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'Black')


# In[17]:


sns.set_context("poster", font_scale = .8)
plt.figure(figsize = (30, 10))
ax = df["Furnishing Status"].value_counts().plot(kind = 'bar', color = "Pink", rot = 0)

for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'Black')


# In[15]:


#Pie Plot on Cities to check the distribution

plt.figure(figsize = (20, 8))
counts = df["City"].value_counts()
explode = (0, 0, 0, 0, 0, 0.1)
colors = ['#FF1E00', '#A66CFF', '#EAE509', '#D61C4E', '#3CCF4E', '#3AB4F2']

counts.plot(kind = 'pie', colors = colors, explode = explode, autopct = '%1.1f%%')
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()


# In[25]:


fig = px.sunburst(df, path=['City','Area Type', 'Furnishing Status', 'Tenant Preferred'], width=900,
    height=900,title='Allotment of flats according to Bachelors/Family/(Bachelors/Family)',color_discrete_sequence=px.colors.cyclical.Phase)
fig.show()


# In[26]:


from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[58]:


X = df.drop(["Rent","Posted On","Floor","Area Locality","Area Type","City","Furnishing Status","Tenant Preferred","Point of Contact"], axis=1)
y = df["Rent"]


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)


# In[60]:


# Scaling the data
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)


# In[ ]:




