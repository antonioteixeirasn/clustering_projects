#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Importando o dataset

dataset = pd.read_csv("Mall_Customers.csv")

# Não possui variável dependendo em problemas de clusteriszação
X = dataset.iloc[:, 3:5].values
dataset.head()


# In[3]:


# Usando o Dendrograma para encontrar o melhor número de clusteres

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()


# In[4]:


# Treinando o modelo Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)


# In[5]:


# Visualizando os Clusters

plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

