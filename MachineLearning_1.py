#!/usr/bin/env python
# coding: utf-8

# In[33]:


#veriyi tanıma
from sklearn.datasets import load_iris
iris=load_iris()


# In[34]:


print(iris.keys())


# In[35]:


print(iris["target_names"])#tahmin etmek istediğimiz çiçeğin türünü gösterir.


# In[36]:


print(iris["data"].shape)#veri setinin yapısını gösterir 150 satır 4 sütun


# In[37]:


print(iris["data"][:3])#ilk 3 çiçeğin değerleri yazıldı


# In[38]:


print(iris["DESCR"])#veri setinin özet bilgilerini verir.


# In[39]:


print(iris["feature_names"])#özelliklerin adını gösterir


# In[40]:


print(iris["target"])#0,1,2 türleri gösteriyor


# #veriyi parçalama

# In[41]:


#traning-eğitim verisi
#test verisi
#x=veri(data) matris(2 boyutlu)
#y=etiket(target) vektör


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_egitim, X_test, y_egitim, y_test=train_test_split(iris["data"],iris["target"],random_state=0)


# In[44]:


print(X_egitim.shape)
print(y_egitim.shape)
#eğitim verileri verilerin %75'ini oluşturuyor


# In[45]:


print(X_test.shape)
print(y_test.shape)
#test verileri verilerin %25'ini oluşturuyor


# #veri ön inceleme

# In[46]:


import pandas as pd


# In[47]:


iris_df=pd.DataFrame(X_egitim, columns=iris.feature_names)


# In[48]:


from pandas.plotting import scatter_matrix


# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


scatter_matrix(iris_df,c=y_egitim, figsize=(15,15), marker="o",hist_kwds={"bins":20}, s=80, alpha=0.8)


# #model oluşturma

# In[51]:


# model oluşturma eğitim verisiyle yapılır.


# In[52]:


from sklearn.neighbors import KNeighborsClassifier


# In[53]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[54]:


knn.fit(X_egitim,y_egitim)


# In[55]:


import numpy as np
X_yeni=np.array([[5,2.9,1,0.2]])
X_yeni.shape


# In[57]:


tahmin=knn.predict(X_yeni)
print("Tahmin sınıfı:",tahmin)
print("Tahmin türü:",iris["target_names"][tahmin])


# #modelin performansı

# In[58]:


y_tahmin=knn.predict(X_test)
print(y_tahmin)


# In[59]:


print(np.mean(y_tahmin==y_test))


# In[60]:


print(knn.score(X_test,y_test))


# In[ ]:




