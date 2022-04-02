#!/usr/bin/env python
# coding: utf-8

# # Self Organizing Maps

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
# http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset.head(15)
# dataset.iloc[:, 1:-1]


# In[3]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
X


# In[4]:


# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# In[5]:


# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
show()


# In[6]:


# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors  = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# In[7]:


# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(0,4)], mappings[(1,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)
for client in frauds:
    print(client[0])


# ---
# # SimpSOM

# In[35]:


#! pip install simpsom
#import pandas as pd
import simpsom as sps
from sklearn.cluster import KMeans
#import numpy as np

net = sps.SOMNet(20, 20, X, PBC=True)
net.train(train_algo='online', epochs=1000)
net.save('filename_weights')
net.nodes_graph(colnum=0)


# In[11]:


net.diff_graph()


# In[ ]:


#Project the datapoints on the new 2D network map.
net.project(X)


# In[ ]:


#Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(X, type='qthresh')	


# In[ ]:




