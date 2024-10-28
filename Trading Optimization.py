#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import ta
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


# In[52]:


df = yf.download('SPY', start = '2020-01-01')


# In[53]:


df['RSI']=ta.rsi(df.Close, length=10)
my_bbands = ta.bbands(df.Close, length=15, std=1.5)
df['ATR']=ta.atr(df.High, df.Low, df.Close, length=7)
df=df.join(my_bbands)
df


# In[54]:


from tqdm import tqdm
tqdm.pandas()
df.reset_index(inplace=True)


# In[55]:


def total_signal(df, current_candle, backcandles):
    if (df.Close[current_candle]<=df['BBL_15_1.5'][current_candle]
        and df.RSI[current_candle]<70
        ):
            return 2
    if (df.Close[current_candle]>=df['BBU_15_1.5'][current_candle]
        and df.RSI[current_candle]>30
        ):
    
            return 1
    return 0
        


# In[56]:


df['TotalSignal'] = df.progress_apply(lambda row: total_signal(df, row.name, 7), axis=1)


# In[57]:


def SIGNAL():
    return df.TotalSignal


# In[58]:


import ta


# In[60]:


class Strat(Strategy):
    
    n1 = 50
    n2 = 100
    slcoef = 1.2 #1.3
    TPSLRatio = 2 # 1.8
    mysize = 0.99
 
    def init(self):
        close = self.data.Close
        self.sma1 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n1)
        self.sma2 = self.I(ta.trend.sma_indicator, pd.Series(close), self.n2)
        
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()
    
    def init(self):
        close=self.data.Close
        self.macd = self.I(ta.trend.macd, pd.Series(close))
        self.macd_signal = self.I(ta.trend.macd_signal, pd.Series(close))
        self.ema_100 = self.I(ta.trend.ema_indicator, pd.Series(close), window=100)
            
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        slatr = self.slcoef*self.data.ATR[-1]
        TPSLRatio = self.TPSLRatio

        if len(self.trades)>0:
            if self.trades[-1].is_long and self.data.RSI[-1]>=90:
                self.trades[-1].close()
            elif self.trades[-1].is_short and self.data.RSI[-1]<=10:
                self.trades[-1].close()
        
        if self.signal1==2 and len(self.trades)==0:
            sl1 = self.data.Close[-1] - slatr
            tp1 = self.data.Close[-1] + slatr*TPSLRatio
            self.buy(sl=sl1, tp=tp1, size=self.mysize)
        
        elif self.signal1==1 and len(self.trades)==0:         
            sl1 = self.data.Close[-1] + slatr
            tp1 = self.data.Close[-1] - slatr*TPSLRatio
            self.sell(sl=sl1, tp=tp1, size=self.mysize)

bt = Backtest(df,Strat, cash = 100000, commission = 0.002)


# In[61]:


bt.run()


# In[62]:


optim = bt.optimize(slcoef=[i/10 for i in range(10, 21)],TPSLRatio=[i/10 for i in range(10, 21)], n1 = range(50,160,10), n2 = range(50,160,10), constraint = lambda x: x.n2 - x.n1 >20, maximize = 'Return [%]')


# In[63]:


optim


# In[68]:





# In[64]:


bt.plot()


# In[ ]:





# In[ ]:




