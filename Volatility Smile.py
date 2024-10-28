#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt


# In[4]:


ticker = yf.Ticker('AAPL')


# In[12]:


ticker.options


# In[13]:


expiration = ticker.options[2]


# In[14]:


option_chain = ticker.option_chain(expiration)


# In[18]:


option_chain


# In[15]:


calls = option_chain.calls


# In[16]:


calls = calls[calls.openInterest >=500]


# In[19]:


calls


# In[17]:


plt.figure(figsize=(10,6))
plt.plot(calls['strike'], calls['impliedVolatility'])


# In[ ]:




