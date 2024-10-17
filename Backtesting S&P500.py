#!/usr/bin/env python
# coding: utf-8

# In[44]:


get_ipython().system('pip install yfinance')
get_ipython().system('pip install ta')
get_ipython().system('pip install backtesting')
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# In[59]:


class SMAcross(Strategy):
    
    n1 = 50
    n2 = 100
    
    def init(self):
        close = self.data.Close
        self.sma1 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n1)
        self.sma2 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n2)
        
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


# In[56]:


df = yf.download('SPY', start = '2020-01-01')


# In[60]:


bt = Backtest(df, SMAcross, cash = 1000000, commission = 0.002, exclusive_orders = True)


# In[61]:


output = bt.run()


# In[64]:


output


# In[65]:


bt.plot()


# In[66]:


optim = bt.optimize(n1 = range(50,160,10), n2 = range(50,160,10), constraint = lambda x: x.n2 - x.n1 >20, maximize = 'Return [%]')

bt.plot()


# In[67]:


optim


# In[ ]:




