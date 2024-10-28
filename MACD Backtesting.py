#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# In[11]:


df = yf.download('SPY', start = '2020-01-01')


# In[20]:


class TradingStrategy(Strategy):
    def init(self):
        close=self.data.Close
        self.macd = self.I(ta.trend.macd, pd.Series(close))
        self.macd_signal = self.I(ta.trend.macd_signal, pd.Series(close))
        self.ema_100 = self.I(ta.trend.ema_indicator, pd.Series(close), window=100)
        
    def next(self):
        price = self.data.Close
        if crossover(self.macd, self.macd_signal) and price > self.ema_100:
            sl = price*0.95 #Stop Loss
            tp = price*1.045 #Target Profit
            self.buy(sl=sl,tp=tp)


# In[21]:


bt = Backtest(df,TradingStrategy, cash = 100000, commission = 0.002)


# In[22]:


output = bt.run()


# In[23]:


output


# In[24]:


bt.plot()


# In[ ]:




