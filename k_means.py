import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt


# Generate Random Data
X = -2 * np.random.rand(100,2)
X1 = 1 + 2* np.random.rand(50,2)
X[50:100, :] = X1

plt.scatter(X[:,0], X[:,1],  s= 50, c = 'b')
plt.show()

# K Means in SCikit Learn

from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

Kmean.cluster_centers_


plt.scatter(X[:,0],X[:,1],s=50, c = 'b')
plt.scatter( -0.90945934, -1.02228219, s = 200, c = 'g', marker='s')
plt.scatter(1.91535798, 1.9685058, s = 200, c = 'r', marker = 's')
plt.show()

Kmean.labels_
test = np.array([-3.0,3.0])
test1 = test.reshape(1,-1)
Kmean.predict(test1)

