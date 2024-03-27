import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class KMeans:
    def __init__(self, k=5, max_iter=50):
        self.k=k
        self.max_iter=max_iter
        self.centroids=None
        self.data=None  

    def load_data(self, data):
        cols=["overall", "age", "potential", "wage_eur", "value_eur"]
        data=data[cols]
        data=data.dropna()  
        data=data[:2500]    
        self.data=StandardScaler().fit_transform(data)
        self.centroids=self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]

    def __calculate_distance(self):
        distance = np.zeros((len(self.data), self.k))
        for i in range(len(self.data)):
            for j in range(self.k):
                distance[i, j] = np.sqrt(np.sum((self.data[i, :] - self.centroids[j, :])**2))
        return distance

    def __plot(self, data, labels, centroids):
        pca=PCA(n_components=2)
        data_2d=pca.fit_transform(data)
        centroids_2d=pca.transform(centroids)
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette=sns.color_palette("bright")[:self.k])
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], color='black')
        plt.show()

    def fit(self):
        for _ in range(self.max_iter):
            distances=self.__calculate_distance()
            labels=np.argmin(distances, axis=1)
            self.__plot(self.data, labels, self.centroids)
            self.centroids=np.array([self.data[labels==i].mean(axis=0) for i in range(self.k)])

data=pd.read_csv("Kmeans\data\players_22.csv")
model= KMeans()
model.load_data(data)
model.fit()