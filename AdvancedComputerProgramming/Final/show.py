import sys
import gc
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import norm, skew, kurtosis  # for some statistics
import IPython
from IPython import display
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl

# %matplotlib inline
# ignore warnings
import warnings
from data5 import train_data, data

#
weights = train_data['happiness'].values
#
sections = [0,1,2,3,4,5]
group_names = ['1','2','3','4','5']
cuts = pd.cut(weights,sections,labels=group_names)
plt.figure(figsize=(10, 8), dpi=70)
plt.xlabel("happiness")
plt.ylabel("quantity")
cuts.value_counts().plot(kind='bar',color='b',alpha = 0.5)




dfData = data.corr()
#plt.subplots(figsize=(9, 9)) # 设置画面大小
plt.figure(figsize=(20, 20), dpi=120)
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
plt.show()

plt.figure(figsize=(10, 8), dpi=70)
sns.countplot('survey_type',hue='happiness',data=train_data,alpha = 0.6)
plt.xlabel("1=urban, 2=rural")

#ax[1].set_title('Sex:happiness')

data['happiness'] = train_data['happiness']
k=data.corr()['happiness'][abs(data.corr()['happiness'])<0.015]
plt.figure(figsize=(8, 12), dpi=70)
plt.ylabel("correlation(abs<0.015)")
plt.xlabel("x(feature)")
k.plot(kind='barh',color=['r','b'],alpha = 0.5)


data['happiness'] = train_data['happiness']
nul = nul[nul[:]>0]
#k=data.corr()['happiness'][abs(data.corr()['happiness'])<0.015]
plt.figure(figsize=(12, 8), dpi=70)
plt.ylabel("missing value")
plt.xlabel("x(feature)")
nul.plot(kind='bar',color=['b'],alpha = 0.5)
