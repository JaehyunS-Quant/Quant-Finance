#!/usr/bin/env python
# coding: utf-8

# In[4]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[24]:


df = yf.download('SPY', start = '2024-01-01')


# In[25]:


df['MA10'] = df['Adj Close'].rolling(10).mean()
df['MA50'] = df['Adj Close'].rolling(50).mean()


# In[26]:


df = df.dropna()


# In[27]:


df = df[['Adj Close', 'MA10', 'MA50']]


# In[28]:


Buy = []
Sell = []

for i in range(len(df)):
    if df.MA10.iloc[i]>df.MA50.iloc[i] and df.MA10.iloc[i-1]<df.MA10.iloc[i-1]:
        Buy.append(i)
    elif df.MA10.iloc[i]<df.MA50.iloc[i] and df.MA10.iloc[i-1]>df.MA50.iloc[i-1]:
        Sell.append(i)


# In[30]:


plt.figure(figsize = (12,10))
plt.plot(df['Adj Close'], label = 'S&P 500', c = 'blue', alpha = 0.5)
plt.plot(df['MA10'], label = 'MA10', c = 'k', alpha = 0.9)
plt.plot(df['MA50'], label = 'MA50', c = 'magenta', alpha = 0.9)
plt.scatter(df.iloc[Buy].index, df.iloc[Buy]['Adj Close'], marker = '^', color = 'g', s=100)
plt.scatter(df.iloc[Sell].index, df.iloc[Sell]['Adj Close'], marker = 'v', color = 'r', s=100)
plt.legend()
plt.show()


# In[ ]:




