"""
Algorithm description:
    for each cluster create a random centroid for each features
    example: for k = 3, cluster  will have 3 values for overall features etc...
    calculate distance between each data point and clusters then assign its label the closest cluster
    finally update the clusters to the mean of all the points in it
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
class KMeans:
    def __init__(self, k=5, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.data=None
    def preprocess(self, data):
        cols=["overall", "age", "potential", "wage_eur", "value_eur"]
        data = data[cols]
        data = data.dropna()
        data = data[:2500]
        scaler = StandardScaler()
        self.data = scaler.fit_transform(data)
        self.centroids = self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]
        print(self.centroids)
    def __init_centroids(self):
        return 
    def __calculate_distance(self):
        distance = np.zeros((len(self.data), self.k))

        for i in range(len(self.data)):
            for j in range(self.k):
                distance[i, j] = np.sqrt(np.sum((self.data[i, :] - self.centroids[j, :])**2))
        return distance
    def __plot(self, data, labels, centroids):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.fit_transform(centroids)
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette=sns.color_palette("bright")[:self.k])
        sns.scatterplot(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color="black")
        plt.draw()
        plt.pause(1)
        plt.clf() 
    def fit(self):
        for _ in range(self.max_iter):
            distance = self.__calculate_distance()
            labels = np.argmin(distance, axis=1)
            self.__plot(self.data, labels, self.centroids)
            self.centroids = np.array([self.data[labels==i].mean(axis=0) for i in range(self.k)])
            print(self.centroids)
        plt.show()    
       




model = KMeans(max_iter=50)
data = pd.read_csv("Kmeans\data\players_22.csv")
model.preprocess(data)
model.fit()