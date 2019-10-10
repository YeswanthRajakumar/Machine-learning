#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np 
import pandas as pd


# In[54]:


ds = pd.read_csv('/home/yeswanth/Desktop/ta.csv')

X = ds[['Pclass','Sex','Age','Fare']]
y = ds.Survived


# In[55]:


X['Age']= X.fillna(X['Age'].median())
X.head()


# In[56]:


from sklearn.preprocessing import LabelEncoder
sex_le = LabelEncoder() 
X['Sex']= sex_le.fit_transform(X['Sex']) 


# In[69]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[64]:


from sklearn.tree import DecisionTreeClassifier
Tree_model = DecisionTreeClassifier()
Tree_model.fit(X_train,y_train)


# In[75]:


X_test.head()


# In[78]:


Tree_model.predict([[2 ,1 ,2,10.5000]])

