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
import Adafruit_DHT
sensor = Adafruit_DHT.DHT11
DHT11_pin = 21

humidity, temperature = Adafruit_DHT.read_retry(sensor, DHT11_pin)
if humidity is not None and temperature is not None:
  print('Temperature={0:0.1f}*C  Humidity={1:0.1f}%'.format(temperature, humidity))
else:
  print('Failed to get reading from the sensor. Try again!')

temperature = round(temperature)
l = ['alapuzha','ernakulam','idukki','kannur','kasargod','kollam','kottayam','kozhikode','malappuram','palakkad','pathanamthitta','tiruvananthapuram','trisuur','wayanad']
val = input("Enter your district: ")
print(val)

d = dict([(y,x+1) for x,y in enumerate(set(l))])



X = [0,4, 3252.4,6.1, temperature,0]

arr = np.array(X)
newarr = arr.reshape(1, -1)


# In[17]:


y_pred = gnb.predict(newarr)


# In[18]:


print(y_pred)


# In[ ]:




