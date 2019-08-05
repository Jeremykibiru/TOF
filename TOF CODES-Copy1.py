#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd


# In[29]:


import numpy as np


# In[30]:


import scipy.optimize as sco


# In[31]:


import matplotlib.pyplot as plt


# In[32]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:





# In[33]:


cd C:/Users/jerem/Desktop/TOF grp work codes


# In[34]:


tofdata1 = pd.read_excel("tofdata2.xlsx", index_col = "date")


# In[35]:


tofdata1.head(n= 4)


# In[36]:


tofdata1 = tofdata1.ffill()


# In[37]:


#####for computing the stock log prices


# In[38]:


tofdata1['BBK log prices'] = np.log(tofdata1['BBK.NR'])
tofdata1['FIRE log prices'] = np.log(tofdata1['FIRE.NR'])

tofdata1['HFCK log prices']= np.log(tofdata1['HFCK.NR'])

tofdata1['ICDC log prices'] = np.log(tofdata1['ICDC.NR'])

tofdata1['JUB log prices'] = np.log(tofdata1['JUB.NR'])

tofdata1['KCB log prices'] =np.log(tofdata1['KCB.NR'])

tofdata1['KPLC log prices'] = np.log(tofdata1['KPLC.NR'])

tofdata1['NMG log prices']= np.log(tofdata1['NMG.NR'])

tofdata1['SCBK log prices'] = np.log(tofdata1['SCBK.NR'])

tofdata1['TOTL log prices'] = np.log(tofdata1['TOTL.NR'])


# In[39]:


tofdata1.head(3)


# In[40]:


tofdata1 = tofdata1.ffill().bfill()


# In[41]:


#####for computing the log returns of the stocks


# In[42]:


tofdata1['BBK log returns'] = np.log(tofdata1['BBK.NR']) - np.log(tofdata1['BBK.NR'].shift(1))
tofdata1['FIRE log returns'] = np.log(tofdata1['FIRE.NR']) - np.log(tofdata1['FIRE.NR'].shift(1))

tofdata1['HFCK log returns'] = np.log(tofdata1['HFCK.NR']) - np.log(tofdata1['HFCK.NR'].shift(1))

tofdata1['JUB log returns'] = np.log(tofdata1['JUB.NR']) - np.log(tofdata1['JUB.NR'].shift(1))

tofdata1['KCB log returns']=  np.log(tofdata1['KCB.NR']) - np.log(tofdata1['KCB.NR'].shift(1))

tofdata1['ICDC log returns'] = np.log(tofdata1['ICDC.NR']) - np.log(tofdata1['ICDC.NR'].shift(1))

tofdata1['KPLC log returns'] = np.log(tofdata1['KPLC.NR']) - np.log(tofdata1['KPLC.NR'].shift(1))

tofdata1['NMG log returns'] = np.log(tofdata1['NMG.NR']) - np.log(tofdata1['NMG.NR'].shift(1))

tofdata1['SCBK log returns'] = np.log(tofdata1['SCBK.NR']) - np.log(tofdata1['SCBK.NR'].shift(1))

tofdata1['TOTL log returns'] = np.log(tofdata1['TOTL.NR']) - np.log(tofdata1['TOTL.NR'].shift(1))


# In[43]:


#####for computing normal returns columns in the tofdata1


# In[44]:


tofdata1['BBK simple returns'] = tofdata1['BBK.NR'].pct_change()
tofdata1['FIRE simple returns'] = tofdata1['FIRE.NR'].pct_change()

tofdata1['HFCK simple returns'] = tofdata1['HFCK.NR'].pct_change()

tofdata1['JUB simple returns'] = tofdata1['JUB.NR'].pct_change()

tofdata1['KCB simple returns'] = tofdata1['KCB.NR'].pct_change()

tofdata1['ICDC simple returns'] = tofdata1['ICDC.NR'].pct_change()

tofdata1['KPLC simple returns'] = tofdata1['KPLC.NR'].pct_change()

tofdata1['NMG simple returns'] = tofdata1['NMG.NR'].pct_change()

tofdata1['SCBK simple returns'] = tofdata1["SCBK.NR"].pct_change()

tofdata1['TOTL simple returns'] = tofdata1['TOTL.NR'].pct_change()


# In[45]:


tofdata1.head(4)


# In[46]:


tofdata1 = tofdata1.ffill().bfill()


# In[ ]:





# In[47]:


######plotting the log prices and simple stock prices


# In[48]:


####(A) plotting the simple stock prices


# In[49]:


import matplotlib.pyplot as plt


# In[50]:


simplepricedataframe = tofdata1[['BBK.NR','FIRE.NR','HFCK.NR','ICDC.NR','JUB.NR','KCB.NR','KPLC.NR','NMG.NR','SCBK.NR','TOTL.NR']].copy()


# In[51]:


simplepricedataframe.head(3)


# In[ ]:





# In[ ]:


#### 2 part A


# In[53]:


########(I)plotting the log stock prices


# In[54]:


logpricedataframe = tofdata1[['BBK log prices','KCB log prices','FIRE log prices','ICDC log prices','HFCK log prices','KPLC log prices','NMG log prices','SCBK log prices','TOTL log prices','JUB log prices']].copy()


# In[55]:


(logpricedataframe/logpricedataframe.iloc[0]*100).plot(figsize=(15,10))


# In[56]:


#######(II)COMPUTING THE SAMPLE COVARIANCE MATRICES FOR THE LOG AND SIMPLE RETURN SERIES


# In[57]:


######variance covariance matrix for the log returns series


# In[58]:


dataframe1 = tofdata1[['BBK log returns','FIRE log returns','KCB log returns','HFCK log returns','NMG log returns','ICDC log returns','JUB log returns','KPLC log returns','SCBK log returns','TOTL log returns']].copy()


# In[ ]:





# In[59]:


dataframe1.head(3)


# In[60]:


dataframe1.cov()


# In[ ]:





# In[61]:


####simple returns variance covariance matrix


# In[62]:


dataframe2 = tofdata1[['BBK simple returns','FIRE simple returns','KCB simple returns','HFCK simple returns','ICDC simple returns','KPLC simple returns','NMG simple returns','SCBK simple returns','TOTL simple returns','JUB simple returns']].copy()


# In[63]:


dataframe2.head(3)


# In[64]:


dataframe2.cov()


# In[65]:


########(II)COMPUTING MEAN OF THE LOG AND SIMPLE RETURN SERIES FOR THE STOCKS


# In[66]:


import statistics


# In[67]:


######sample mean of log returns


# In[258]:


dataframe1.mean()


# In[ ]:





# In[78]:


#####computiing mean return of simmple stock price


# In[253]:


dataframe2.mean()


# In[ ]:





# In[89]:


########(III)SKEWNESS KURTOSIS AND JARQUE-BERA TEST FOR RETURN SERIES


# In[90]:


##KURTOSIS


# In[91]:


###log returns kurtosis


# In[92]:


from scipy.stats import kurtosis


# In[250]:


dataframe1.kurtosis()


# In[103]:


###simple return series kurtosis


# In[251]:


dataframe2.kurtosis()


# In[114]:


######SKEWNESS


# In[115]:


###log return series skewness


# In[116]:


from scipy.stats import skew


# In[248]:


dataframe1.skew()


# In[127]:


#####simple return series skewness


# In[246]:


dataframe2.skew()


# In[138]:


########JARQUE-BERA TEST


# In[139]:


####for log returns series


# In[140]:


from statsmodels.stats.stattools import jarque_bera


# In[141]:


jarque_bera(dataframe1['BBK log returns'])


# In[142]:


jarque_bera(dataframe1['SCBK log returns'])


# In[143]:


jarque_bera(dataframe1['KCB log returns'])


# In[144]:


jarque_bera(dataframe1['ICDC log returns'])


# In[145]:


jarque_bera(dataframe1['KPLC log returns'])


# In[146]:


jarque_bera(dataframe1['NMG log returns'])


# In[147]:


jarque_bera(dataframe1['TOTL log returns'])


# In[148]:


jarque_bera(dataframe1['JUB log returns'])


# In[149]:


jarque_bera(dataframe1['FIRE log returns'])


# In[150]:


jarque_bera(dataframe1['HFCK log returns'])


# In[151]:


#####for simple return series


# In[152]:


jarque_bera(dataframe2['BBK simple returns'])


# In[153]:


jarque_bera(dataframe2['TOTL simple returns'])


# In[154]:


jarque_bera(dataframe2['SCBK simple returns'])


# In[155]:


jarque_bera(dataframe2['FIRE simple returns'])


# In[156]:


jarque_bera(dataframe2['KPLC simple returns'])


# In[157]:


jarque_bera(dataframe2['NMG simple returns'])


# In[158]:


jarque_bera(dataframe2['JUB simple returns'])


# In[159]:


jarque_bera(dataframe2['ICDC simple returns'])


# In[160]:


jarque_bera(dataframe2['KCB simple returns'])


# In[161]:


jarque_bera(dataframe2['HFCK simple returns'])


# In[ ]:





# In[162]:


################### 2 part(B)


# In[ ]:


### B (I)


# In[163]:


monthlydata = pd.read_excel("monthlydata.xlsx", index_col = "Exchange Date")


# In[164]:


monthlydata.head()


# In[165]:


monthlydata = monthlydata.ffill()


# In[166]:


ret = (monthlydata.pct_change())


# In[167]:


ret.head(3)


# In[168]:


ret.dropna(inplace = True)


# In[269]:


z=ret.mean()*12
z


# In[170]:


zc=ret.cov()*12


# In[171]:


np.random.seed(12)
num_ports = 6000
all_weights = np.zeros((num_ports, len(monthlydata.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    # Weights
    weights = np.array(np.random.random(10))
    weights = weights/np.sum(weights)
    
    # Save weights
    all_weights[x,:] = weights
    
    # Expected return
    ret_arr[x] = np.sum( (z * weights))
    
    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(zc, weights)))
    
    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]


# In[172]:


print('max sharpe ratio in the array:{}'.format(sharpe_arr.max()))
print("its location in the array:{}".format(sharpe_arr.argmax()))


# In[173]:


print(all_weights[1330,:])


# In[174]:


max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]


# In[175]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
plt.show()


# In[176]:


def get_ret_vol_sr(weights):
    weights = np.array(weights)
    rets = np.sum(z * weights) 
    vols = np.sqrt(np.dot(weights.T, np.dot(zc, weights)))
    sr = rets/vols
    return np.array([rets, vols, sr])

def neg_sharpe(weights):
# the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1


# In[177]:


cons = ({'type': 'eq', 'fun':check_sum})

weightguesses = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)


# In[178]:


opt_results = sco.minimize(neg_sharpe,weightguesses,method = 'SLSQP', constraints = cons)


# In[179]:


####hence our portfolio is unnconstrained thereby allowing for shortselling due to presence of negative weights.


# In[180]:


opt_results['x'].round(6)


# In[181]:


get_ret_vol_sr(opt_results.x)


# In[182]:


frontier_y = np.linspace(0.04,0.6,100)


# In[183]:


def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]


# In[184]:


frontier_x = []

for possible_return in frontier_y:
    cons = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    result = sco.minimize(minimize_volatility,weightguesses,method='SLSQP', constraints=cons)
    frontier_x.append(result['fun'])


# In[185]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_x,frontier_y, 'r-', linewidth=3)
plt.show()


# In[ ]:





# In[ ]:


##### B (II)


# In[186]:


tbilldata = pd.read_excel("tbill rates.xlsx", index_col = "DATES")


# In[187]:


tbilldata.head(3)


# In[188]:


Rf= statistics.mean(tbilldata['tbill'])


# In[189]:


Rf


# In[190]:


import scipy.interpolate as sci


# In[240]:


ind = np.argmin(frontier_x)
evols = frontier_x[ind:]
erets = frontier_y[ind:]

lck = sci.splrep(evols, erets)

def f(x):
        return sci.splev(x, lck, der=0)
## Efficient frontier function (splines approximation)
def df(x):
    return sci.splev(x, lck, der=1)

### First derivative of efficient frontier function.

def equations(p, rf = -0.8):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.001, 0.5, 0.15])

opt

plt.figure(figsize=(12, 9))
###plt.scatter(vol_arr, ret_arr, c=(ret_arr - -0.5) / vol_arr, marker='o')
####random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(-0.8, 0.4)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
# capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='-', lw=2.0)
plt.axvline(0, color='k', ls='-', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
####plt.colorbar(label='Sharpe ratio')


# In[196]:


####due to the unconstraiined nature of the portfolio weights that allow for shortselling. the tangency portfolio is got at the 
#### global minimum variance and the risk free rate would have to be negative so as to allow for tangency which is not feasible
###we thus concluded that the investor`s optimal investment would yield him a retun of 0.1


# In[197]:


eretsw = erets.flat


# In[198]:


np.where(eretsw == 0.10222222222222221)


# In[199]:


print(all_weights[2,:])


# In[ ]:





# In[ ]:


#### B(III) 95% confidence intervall for boot strapped sharpe ratios


# In[239]:


bootarr = np.random.choice(sharpe_arr, size = 1000)

import pandas as pd

bootdf = pd.DataFrame(bootarr)


# In[236]:


perc025 = bootdf.quantile(0.025)

perc0975 = bootdf.quantile(0.975)


# In[237]:


perc025


# In[238]:


perc0975


# In[ ]:





# In[ ]:





# In[200]:


####C


# In[201]:


cons1 = ({'type': 'eq', 'fun':check_sum})
bnds1 = ((0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5))
weightguesses = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)


# In[202]:


opt_results2 = sco.minimize(neg_sharpe,weightguesses,method = 'SLSQP', bounds = bnds1, constraints = cons1)


# In[203]:


frontier_y1 = np.linspace(0.01,0.45,100)


# In[204]:


frontier_x1 = []

for possible_return in frontier_y1:
    cons1 = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    resultb = sco.minimize(minimize_volatility,weightguesses,method='SLSQP', bounds=bnds1, constraints=cons1)
    frontier_x1.append(resultb['fun'])


# In[205]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_x1,frontier_y1, 'r-', linewidth=3)
plt.show()


# In[206]:


ind1 = np.argmin(frontier_x1)
evols1 = frontier_x1[ind1:]
erets1 = frontier_y1[ind1:]


# In[207]:


lck1 = sci.splrep(evols1, erets1)


# In[208]:


def f(x):
        return sci.splev(x, lck1, der=0)
## Efficient frontier function (splines approximation)
def df(x):
    return sci.splev(x, lck1, der=1)

### First derivative of efficient frontier function.


# In[209]:


def equations1(p, rf = Rf):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


# In[210]:


opt1 = sco.fsolve(equations1, [0.001, 0.5, 0.15])


# In[211]:


opt1


# In[212]:


plt.figure(figsize=(12, 9))
####plt.scatter(vol_arr, ret_arr,
#####c=(ret_arr - rf1) / vol_arr, marker='o')
####random portfolio composition
plt.plot(evols1, erets1, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(Rf, 0.5)
plt.plot(cx, opt1[0] + opt1[1] * cx, lw=1.5)
# capital market line
plt.plot(opt1[2], f(opt1[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='-', lw=2.0)
plt.axvline(0, color='k', ls='-', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
####plt.colorbar(label='Sharpe ratio')


# In[213]:


eretsz = erets1.flat


# In[214]:


##the tangecy portfolio return is approximately 0.32555555555555554 and are located at position 50 in the expected returns array


# In[215]:


np.where(eretsz ==  0.32555555555555554)


# In[216]:


print(all_weights[50,:])


# In[217]:


####the above weights are for the 10  assets at the tangecy portfolio


# In[ ]:





# In[218]:


####D


# In[219]:


cons2 = ({'type': 'eq', 'fun':check_sum})
bnds2 = ((0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5),(0,0.5))
weightguesses = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)


# In[220]:


opt2 = sco.minimize(neg_sharpe,weightguesses,method = 'SLSQP', bounds = bnds2, constraints = cons2)


# In[221]:


frontier_y2= np.linspace(0.01,0.45,100)


# In[222]:


frontier_x2 = []

for possible_return in frontier_y2:
    cons2 = ({'type':'eq', 'fun':check_sum},
            {'type':'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})
    
    resultc = sco.minimize(minimize_volatility,weightguesses,method='SLSQP', bounds=bnds2, constraints=cons2)
    frontier_x2.append(resultc['fun'])


# In[223]:


plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_x2,frontier_y2, 'r-', linewidth=3)
plt.show()


# In[224]:


ind2 = np.argmin(frontier_x2)
evols2 = frontier_x2[ind2:]
erets2 = frontier_y2[ind2:]


# In[225]:


lck2 = sci.splrep(evols2, erets2)

def f(x):
        return sci.splev(x, lck2, der=0)
## Efficient frontier function (splines approximation)
def df(x):
    return sci.splev(x, lck2, der=1)

### First derivative of efficient frontier function.


# In[226]:


def equations2(p1, rf1 = 0.07747707):
    eq1 = rf1 - p1[0]
    eq2 = rf1 + p1[1] * p1[2] - f(p1[2])
    eq3 = p1[1] - df(p1[2])
    return eq1, eq2, eq3


# In[227]:


opt2 = sco.fsolve(equations2, [0.001, 0.5, 0.15])

opt2


# In[228]:


plt.figure(figsize=(12, 9))
####plt.scatter(vol_arr, ret_arr,
#####c=(ret_arr - rf1) / vol_arr, marker='o')
####random portfolio composition
plt.plot(evols2, erets2, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(0.07747707, 0.5)
plt.plot(cx, opt2[0] + opt2[1] * cx, lw=1.5)
# capital market line
plt.plot(opt2[2], f(opt2[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='-', lw=2.0)
plt.axvline(0, color='k', ls='-', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
####plt.colorbar(label='Sharpe ratio')


# In[229]:


## the investors target return 0.07 is not located on the expected returns array but the clossest return to that is 0.1033333333
## it is located in array 0


# In[230]:


eretsy = erets2.flat


# In[231]:


eretsy


# In[263]:


np.where(eretsy == 0.10333333333333333)


# In[264]:


print(all_weights[0,:])


# In[234]:


###the above are the weights if he expeccts a minimum return of 0.07


# In[267]:


np.where(eretsw ==0.6)


# In[268]:


list(eretsw)
print(all_weights[90,:])


# In[ ]:




