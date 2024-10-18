#!/usr/bin/env python
# coding: utf-8

# In[27]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta


# In[31]:


class Backtest:
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = yf.download(self.symbol, start = '2019-01-01')
        if self.df.empty:
            print('No Data Pulled')
        else:
            self.calc_indicators()
            self.generate_signals()
            self.loop_it()
            self.profit = self.calc_profit()
            self.cumul_profit = (self.profit + 1).prod() - 1
            
    def calc_indicators(self):
        self.df['MA20'] = self.df.Close.rolling(20).mean()
        self.df['Vol'] = self.df.Close.rolling(20).std()
        self.df['Upper_bb'] = self.df.MA20 + (2*self.df.Vol)
        self.df['Lower_bb'] = self.df.MA20 - (2*self.df.Vol)
        self.df['rsi'] = ta.momentum.rsi(self.df.Close, window = 6)
        self.df.dropna(inplace = True)
        
    def generate_signals(self):
        conditions = [(self.df.rsi <30) & (self.df.Close < self.df.Lower_bb), (self.df.rsi >70) & (self.df.Close >self.df.Upper_bb)]
        choices = ['Buy', 'Sell']
        self.df['signal'] = np.select(conditions, choices)
        self.df.signal = self.df.signal.shift()
        self.df.dropna(inplace = True)
        
    def loop_it(self):
        position = False
        buydates, selldates = [], []
        
        for index, row in self.df.iterrows():
            if not position and row['signal'] == 'Buy':
                position = True
                buydates.append(index)
                
            if position and row['signal'] == 'Sell':
                position = False
                selldates.append(index)
                
        self.buy_arr = self.df.loc[buydates].Open
        self.sell_arr = self.df.loc[selldates].Open
        
    def calc_profit(self):
        if self.buy_arr.index[-1] > self.sell_arr.index[-1]:
            self.buy_arr = self.buy_arr[:-1]
        return (self.sell_arr.values - self.buy_arr.values)/self.buy_arr.values
    
    def plot_chart(self):
        plt.figure(figsize = (10,5))
        plt.plot(self.df.Close)
        plt.scatter(self.buy_arr.index, self.buy_arr.values, marker = '^', c = 'g')
        plt.scatter(self.sell_arr.index, self.sell_arr.values, marker = 'v', c = 'r')


# In[38]:


instance.buy_arr.values


# In[32]:


instance = Backtest('SPY')


# In[33]:


instance.plot_chart()


# In[ ]:




