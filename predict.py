#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle as cPickle

# load it again
with open('GNB_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)


# In[6]:



# importing library
import numpy as np


# In[ ]:





# In[3]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

gnb = gnb_loaded

gnb


# In[16]:


X = [0,4, 3252.4,6.1, 25,0]

arr = np.array(X)
newarr = arr.reshape(1, -1)


# In[17]:


y_pred = gnb.predict(newarr)


# In[18]:


y_pred


# In[ ]:




