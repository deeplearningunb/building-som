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


# In[141]:


# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(0,0)], mappings[(0,9)], mappings[(4,9)]), axis = 0)
frauds = sc.inverse_transform(frauds)
for client in frauds:
    print(int(client[0]))


# ---
# # SimpSOM

# In[8]:


# ! pip install SimpSOM
import pandas as pd
import simpsom as sps
from sklearn.cluster import KMeans
import numpy as np

net = sps.SOMNet(20, 20, X, PBC=True, init='random')
net.train(train_algo='online', start_learning_rate=0.5, epochs=100)
net.save('simpsom_weights')
net.nodes_graph(colnum=0, colname='Customer ID')


# In[9]:


sps_diff = net.diff_graph(returns=True)


# In[10]:


#Project the datapoints on the new 2D network map.
sps_bmu = net.project(X, labels=y, show=True)


# In[11]:


#Cluster the datapoints according to the Quality Threshold algorithm.
sps_clusters = net.cluster(X, clus_type='qthresh', show=True)


# In[85]:


nodes = net.node_list
print('length nodes: ', len(nodes))

nodes_pos_x = [i.pos[0] for i in nodes]
nodes_pos_y = [i.pos[1] for i in nodes]

#print(nodes_pos)

print('length nodes_pos_x: ', len(nodes_pos_x))
print('length nodes_pos_y: ', len(nodes_pos_y))

#for i in nodes:
#    print(i.pos)


# In[20]:


sps_diff


# In[21]:


sps_bmu


# In[22]:


sps_clusters


# In[115]:


g = []
gi = []
# [nodes_pos_x[i], nodes_pos_y[i], i, bmui, y[bmui]]
gc = []

# find nodes by their differences
for i, e in enumerate(sps_diff):
    if e > 0.5:
        g.append(e)
        gi.append(i)
        for bmui, bmu in enumerate(sps_bmu):
            if bmu == [nodes_pos_x[i], nodes_pos_y[i]] and [nodes_pos_x[i], nodes_pos_y[i], i, bmui, y[bmui]] not in gc:
                gc.append([nodes_pos_x[i], nodes_pos_y[i], i, bmui, y[bmui]])

print(len(g), g)
print(len(gi), gi)
print(len(gc), gc)


# In[133]:


# find the answers for the similar clients
answers = {}
for x in gc:
    if (x[0], x[1]) in answers:
        answers[(x[0], x[1])].append([x[3], x[4]])
    else:
        answers[(x[0], x[1])] = []

print('answers', answers)        
        
filtered_answers = {}
# find if people have different outcomes
for k in answers.keys():
    # get the outcomes f
    v = [x[1] for x in answers[k]]
    print(v)
    diff = v.count(v[0]) != len(v)
    if (diff):
        filtered_answers[k] = answers[k]

filtered_answers


# In[140]:


# get the possible frauds
frauds = []
for k in filtered_answers.keys():
    for v in filtered_answers[k]:
        frauds.append(X[v[0]]) 

frauds = sc.inverse_transform(frauds)

for f in frauds:
    print(int(f[0]))


# In[ ]:




