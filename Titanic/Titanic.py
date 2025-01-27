#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r"C:\Users\Usuario\Desktop\Projects\Titanic\train.csv")
df["Age"].fillna(0,inplace = True)
test = pd.read_csv(r"C:\Users\Usuario\Desktop\Projects\Titanic\test.csv")
test["Age"].fillna(0,inplace = True)


# In[4]:


df.head(20)


# In[5]:


df.describe()


# In[6]:


df_num = df[["Age","SibSp","Parch","Fare"]]
df_cat = df[["Survived","Pclass","Sex","Ticket","Cabin","Embarked"]]


# In[7]:


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[8]:


print(df_num.corr())
sns.heatmap(df_num.corr())


# In[9]:


pd.pivot_table(df, index = "Survived", values = ["Age","SibSp","Parch","Fare"])# mitjana de edad segons si han mort o no...


# In[10]:


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i) #index te los indexa pa que salgan con el titulillo abajo
    plt.show()#.value_counts() te cuenta cuantas veces se repite un valor


# In[11]:


print(pd.pivot_table(df, index = "Survived", columns = "Pclass", values = "Ticket", aggfunc = "count"))# mitjana de edad segons si han mort o no...
print(pd.pivot_table(df, index = "Survived", columns = "Sex", values = "Ticket", aggfunc = "count"))# mitjana de edad segons si han mort o no...
print(pd.pivot_table(df, index = "Survived", columns = "Embarked", values = "Ticket", aggfunc = "count"))# mitjana de edad segons si han mort o no...


# In[12]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

cols = ["Sex","Embarked"]

for col in cols:
    df[col] = le.fit_transform(df[col])
    test[col] = le.transform(test[col])
    print(le.classes_)


# In[13]:


df.head(20)


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df  = df.drop(["Name","Cabin","Ticket"], axis=1)
test = test.drop(["Name","Cabin","Ticket"], axis=1)
Y = df["Survived"]
X = df.drop("Survived", axis = 1)

#aixo el que fa es crear un dataset de entrenament que s'entrena
#en las X poses el que vols mirar i en les Y poses la resposta que tenen els parametres x (que es 0 o 1)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size = 0.2, random_state = 42)
df.head(10)


# In[15]:


#creem objecte de regresio logistica i amb fit entrenemn el model
clf = LogisticRegression(random_state = 0, max_iter = 1000).fit(X_train,Y_train)
#fem prediccions dels valors Y a partir dels valors X del test( que li hem dit X_val pero es X_test)
predictions = clf.predict(X_val)


# In[16]:


predictions  = clf.predict(X_val)

#Compara les prediccions amb les dades reals del test
from sklearn.metrics import accuracy_score
accuracy_score(Y_val,predictions)


# In[17]:


clf.predict_proba(X_val)#calcula la probabilitat de que els passatgers NO sobrevisquin
clf.predict(23)# et prediu si el passatger 23 sobreviurà o no


# In[ ]:


submission_preds = clf.predict(test)


# In[ ]:


data = pd.DataFrame({"PassangerId":test_ids.values, "Survived": submission_preds})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




