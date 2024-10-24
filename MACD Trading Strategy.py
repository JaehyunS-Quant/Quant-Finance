#!/usr/bin/env python
# coding: utf-8

# In[15]:


import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


# In[38]:


df = yf.download('SPY', start = '2024-01-01')


# In[50]:


df


# In[39]:


def MACD(df):
    df['EMA12'] = df.Close.ewm(span=12).mean()
    df['EMA26'] = df.Close.ewm(span=26).mean()
    df['EMA200'] = df.Close.ewm(span=200).mean()
    df['MACD'] = df.EMA12 - df.EMA26
    df['Signal'] = df.MACD.ewm(span=9).mean()
    print('Indicators Added')


# In[40]:


MACD(df)


# In[41]:


plt.plot(df.MACD, label='MACD', color='green')
plt.plot(df.Signal, label='Signal', color='red')
plt.legend()
plt.show()


# In[51]:


Buy, Sell = [], []

for i in range(2,len(df)):
    if df.MACD.iloc[i]>df.Signal.iloc[i] and df.MACD.iloc[i-1]<df.Signal.iloc[i-1]:
        Buy.append(i)
    elif df.MACD.iloc[i]<df.Signal.iloc[i] and df.MACD.iloc[i-1]>df.Signal.iloc[i-1]:
        Sell.append(i)


# In[52]:


plt.scatter(df.iloc[Buy].index, df.iloc[Buy].Close, marker='^', color='green')
plt.scatter(df.iloc[Sell].index, df.iloc[Sell].Close, marker='v', color='red')
plt.plot(df.EMA200, label='EMA200', color='blue')
plt.plot(df.Close, label='S&P500 Close', color='k')
plt.legend()
plt.show()

