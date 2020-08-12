#!/usr/bin/env python
# coding: utf-8

# In[2]:


pwd


# In[3]:


cd /Users/kozlo/Downloads


# In[4]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
waw = pd.read_csv("gios-pjp-data(2).csv", index_col=0, sep=";")
waw.head()


# In[5]:


waw.dtypes


# In[6]:


cols = ['Warszawa-Targówek - pył zawieszony PM10']
waw[cols] = waw[cols].replace(',','.', regex=True).astype(float)


# In[7]:


waw.describe()


# In[8]:


waw.isna().sum()


# In[36]:


series_value = waw.values


# In[37]:


type(series_value)


# In[33]:


waw.size


# In[34]:


waw.tail()


# In[ ]:





# In[9]:


waw['Warszawa-Targówek - pył zawieszony PM10'].fillna(waw['Warszawa-Targówek - pył zawieszony PM10'].mean(), inplace=True)


# In[10]:


waw['Warszawa-Targówek - pył zawieszony PM10'].plot()
plt.xlabel('Czas pomiaru')
plt.ylabel(cols[0])
plt.legend()
plt.show()


# In[11]:


waw.hist(figsize=(10, 6))
plt.show()


# In[12]:


from pandas.plotting import lag_plot
 
lag_plot(waw['Warszawa-Targówek - pył zawieszony PM10'], lag=1)
plt.title('Warszawa-Targówek - pył zawieszony PM10')


# In[13]:


from pandas.plotting import autocorrelation_plot
 
autocorrelation_plot(waw['Warszawa-Targówek - pył zawieszony PM10'])
plt.title('Warszawa-Targówek - pył zawieszony PM10')


# In[42]:


waw_test = waw[1:]


# In[43]:


waw_test.head()


# In[44]:


waw_mean = waw.rolling(window = 20).mean()


# In[45]:


waw.plot()
waw_mean.plot()


# In[88]:


waw = waw[0:365]


# In[89]:


waw.describe()


# In[77]:


value = pd.DataFrame(series_value)


# In[78]:


PM10_df = pd.concat([value,value.shift(1)], axis=1)


# In[79]:


PM10_df.head()


# In[90]:


waw_test = waw_test[0:364]


# In[91]:


waw_test.tail()


# In[ ]:





# In[80]:


PM10_df.columns = ['Actual_waw','Forecast_waw'] 


# In[92]:


PM10_df.head()


# In[82]:


from sklearn.metrics import mean_squared_error


# In[83]:


waw_test = PM10_df[1:]


# In[84]:


waw_test.head()


# In[85]:


waw_error = mean_squared_error(waw_test.Actual_waw,waw_test.Forecast_waw)


# In[93]:


waw_error


# In[94]:


np.sqrt(waw_error)


# In[96]:


ARIMA - Autoregressive (p) Integrated (d) Moving Average (q)


# In[99]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[100]:


#plot acfis to identify parameter Q
#ARIMA (p,d,q)
plot_acf(waw)


# In[162]:


plot_pacf(waw) #to identify the value of P


# In[163]:


waw.size


# In[164]:


waw_train = waw[0:380]
waw_test = waw[0:300]


# In[165]:


waw_train.size


# In[166]:


waw_test.size


# In[151]:


from statsmodels.tsa.arima_model import ARIMA


# In[152]:


waw_model = ARIMA(waw_train, order=(2,0,3))


# In[153]:


waw_model_fit = waw_model.fit()


# In[154]:


waw_model_fit.aic


# In[167]:


waw_forecast = waw_model_fit.forecast(steps = 91)[0]


# In[168]:


waw_forecast


# In[169]:


waw_test


# In[170]:


np.sqrt(mean_squared_error(waw_test,waw_forecast))

