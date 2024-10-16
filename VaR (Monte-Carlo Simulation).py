#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install yfinance')
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm


# In[7]:


years = 15

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)


# In[11]:


tickers = ['SPY', 'GLD', 'QQQ']


# In[12]:


adj_close_df = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start = startDate, end = endDate)
    adj_close_df[ticker] = data['Adj Close']
    


# In[30]:


log_returns = np.log(adj_close_df/adj_close_df.shift(1))
log_returns = log_returns.dropna()
print(log_returns)


# In[14]:


def expected_returns(weights, log_returns):
    return np.sum(log_returns.mean()*weights)


# In[15]:


def standard_deviation (weights, cov_matrix):
    variance = weights.T@cov_matrix@weights
    return np.sqrt(variance)


# In[16]:


cov_matrix = log_returns.cov()


# In[17]:


portfolio_value = 300000
weights = np.array([1/len(tickers)]*len(tickers))
portfolio_expected_return = expected_returns(weights, log_returns)
portfolio_std_dev = standard_deviation (weights, cov_matrix)


# In[18]:


def random_z_score():
    return np.random.normal(0,1)


# In[19]:


days = 10

def scenario_gain_loss(portfolio_value,portfolio_std_dev,z_score, days ):
    return portfolio_value*portfolio_expected_return*days + portfolio_value*portfolio_std_dev*z_score*np.sqrt(days)


# In[23]:


simulations = 10000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value,portfolio_std_dev,z_score, days ))


# In[28]:


confidence_interval = 0.99
VaR = -np.percentile(scenarioReturn, 100*(1-confidence_interval))
print(VaR)


# In[29]:


plt.hist(scenarioReturn, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()


# In[ ]:




