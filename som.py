# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
#pcolor(som.distance_map())
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
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

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(7,3)], mappings[(7,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)
# change the float format to %.3f

# Finding frauds with SimpSOM
import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
import numpy as np

net = sps.somNet(20, 20, X, PBC=True)
net.train(0.5, 100)

net.save('filename_weights')
net.nodes_graph(colnum=0)

net.diff_graph()

## Projecting the data points on the new 2D network map ##

prj = np.array(net.project(X))
plt.scatter(prj.T[0],prj.T[1])
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(prj)
dataset["clusters"]=kmeans.labels_

## SOM combined with k means gives me cluster 0 is related to fraud. ##

dataset[dataset["clusters"]==0].head(20) # First 20 frauds