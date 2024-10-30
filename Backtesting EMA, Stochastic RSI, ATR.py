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


# In[9]:


df = yf.download('SPY', start='2024-01-01')


# In[15]:


class TradePro(Strategy):
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        self.stoch_k = self.I(ta.momentum.stochrsi_k, pd.Series(close))
        self.stoch_d = self.I(ta.momentum.stochrsi_d, pd.Series(close))
        self.EMA_8 = self.I(ta.trend.ema_indicator, pd.Series(close), window=8)
        self.EMA_14 = self.I(ta.trend.ema_indicator, pd.Series(close), window=14)
        self.EMA_50 = self.I(ta.trend.ema_indicator, pd.Series(close), window=50)
        self.atr = self.I(ta.volatility.average_true_range, pd.Series(high), pd.Series(low), pd.Series(close))
        
    def next(self):
        price = self.data.Close
        if (crossover(self.stoch_k, self.stoch_d) and price > self.EMA_8 and self.EMA_8 > self.EMA_14 and self.EMA_14 > self.EMA_50):
            sl = price - self.atr * 3
            tp = price + self.atr * 2
            self.buy(sl = sl, tp = tp)
        


# In[16]:


bt = Backtest(df, TradePro, cash = 10000, exclusive_orders = True)


# In[17]:


output = bt.run()


# In[18]:


output


# In[19]:


bt.plot()


# In[ ]:




