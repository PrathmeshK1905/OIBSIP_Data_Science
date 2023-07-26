#!/usr/bin/env python
# coding: utf-8

# # Loading Scikit-learn Dataset

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()


# In[3]:


iris


# In[4]:


iris.data


# In[5]:


iris.target


# In[6]:


iris.target_names


# In[7]:


iris.feature_names


# In[8]:


x = iris.data
y = iris.target


# # Splitting Data into Training and Testing

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 4)
x_train


# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# In[12]:


knn = KNeighborsClassifier(n_neighbors = 5)


# In[13]:


knn.fit(x_train, y_train)


# In[14]:


y_pred = knn.predict(x_test)
y_pred


# In[15]:


y_test


# # Accuracy

# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:




