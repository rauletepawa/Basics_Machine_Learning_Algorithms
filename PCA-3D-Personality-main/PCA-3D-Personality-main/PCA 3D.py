#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')
import codecs
import os
from mpl_toolkits import mplot3d
from bioinfokit.visuz import cluster


# In[45]:


#fem llista per titular les columnes
lst = list(range(1,30+1))
llista = []
llista.append("target")
for i in lst:
    llista.append(str(i))
print(llista)


# In[46]:


with open(r"C:\Users\Usuario\Desktop\PAUP\dades.txt","r") as data_file:
    dades = data_file.read()
    print(dades)


# In[47]:


# loading dataset into Pandas DataFrame
df = pd.read_csv(r"C:\Users\Usuario\Desktop\PAUP\dades.txt", encoding="ISO-8859-1",
                   sep = "\t", names = llista, keep_default_na=False)#read_csv llegeix taula de l'arxiu de la direcció indicada
                                                                    #encoding es per a que llegeixi be els valors i no doni encoding error
    
df.head()


# In[48]:


features = llista[1:]#separem dades dels noms de les persones
x = df.loc[:, features].values
print(x)


# In[49]:


y = df.loc[:,['target']].values
print(y)
llista_noms = []#llista normal amb tots el
for i in y:
    for a in i:
        llista_noms.append(a)


# In[50]:


x = StandardScaler().fit_transform(x)#aquesta funció d'aqui estandaritza els valors per a que el pca es pugui fer be!!


# In[51]:


print(x)


# In[52]:



pca = PCA(n_components=3)# seleccionem que volem un pca de 2 components principals
principalComponents = pca.fit_transform(x)# entrem les dades als components
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PCA1', 'PCA2','PCA3'])#fem un dataframe amb PC1 i PC2 i PC3


# In[53]:


principalDf.head(5)


# In[54]:


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)#juntem les taules del component 1,2 amb els noms
finalDf.head(5)


# In[56]:


print(principalDf)


# In[58]:


nmp=principalDf.to_numpy()# passa el dataframe a una matriu
print(nmp)


# In[67]:


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

 
m=nmp # m is an array of (x,y,z) coordinate triplets
fig=plt.figure(figsize=(20,20))


ax = fig.add_subplot(projection='3d')

for i in range(len(m)): #plot each point + it's index as text above
    ax.scatter(m[i,0],m[i,1],m[i,2],color='b') 
    ax.text(m[i,0],m[i,1],m[i,2],  '%s' % (llista_noms[i]), size=20, zorder=1,  
    color='k') #Llista_noms[i] indica el nom de cada punt!!!!!

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
pyplot.show()




print(pca.explained_variance_ratio_)





