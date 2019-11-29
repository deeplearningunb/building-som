# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Interpretação do Mapa: A partir do mapa que foi gerado percebemos que os pokemons lendarios se agrupam em regiões mais escuras e da mesma forma
os pokemons que não são lendários também se encontram nessas mesmas regiões, ou seja, conseguimos sintetizar as informações e descobrir semelhanças em níveis de força e atributos, independente da sua classe (lendários ou não).
Demonstrando assim que é possível os pokémons ditos "comuns" serem capazes de batalhar e vencer pokemons lendarios.

"""
# Transformando dados Categóricos
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() #instanciando objeto

# Importing the dataset
dataset = pd.read_csv('pokemon.csv')
X = dataset.iloc[:, 4:11].values # Escolhendo principais informações das colunas
y = dataset.iloc[:, -1].values # Ultima coluna

y = label_encoder.fit_transform(y)

#print (X)
#print (y)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#print (X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 7, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
#pcolor(som.distance_map())
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'b']
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
frauds = np.concatenate((mappings[(7,3)], mappings[(7,4)]), axis = 0)
frauds = sc.inverse_transform(frauds)
# change the float format to %.3f