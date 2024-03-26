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
class KMeans:
    def __init__(self, k=2, max_iter=50):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.data=None
    def preprocess(self):
        self.data = self.data.dropna()

    def load_data(self, data):
        cols = ["overall", "age", "potential", "wage_eur", "value_eur"]
        data = data[cols]
        data = data[:2500]
        scaler = StandardScaler().set_output(transform="pandas")
        data = scaler.fit_transform(data)
        self.data = data
        print(self.data.describe())
        self.centroids = self.data.sample(n=self.k)
        self.centroids.index = [i for i in range(self.k)]
    def __calculate_distance(self, distance):
        for i in self.data.index:
                for j in range(self.k):
                    distance.loc[i, j] = np.sqrt(np.sum(((self.data.loc[i, :] - self.centroids.loc[j, :])**2)))
        return distance
    def __plot(self, data, labels, centroids):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.fit_transform(centroids)
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels, palette=sns.color_palette("bright")[:self.k])
        sns.scatterplot(x=centroids_2d[:, 0], y=centroids_2d[:, 1], color="black")
        plt.draw()
        plt.pause(0.1)
        plt.clf() 
    def fit(self):
        distance = pd.DataFrame(columns=[i for i in range(self.k)], index = [i for i in self.data.index])
        for _ in range(self.max_iter):
            self.__calculate_distance(distance)      
            labels = distance.idxmin(axis=1)
            self.__plot(self.data, labels, self.centroids)
            self.centroids = self.data.groupby(labels).mean()
            print(self.centroids)
        plt.show()       
       




model = KMeans()
data = pd.read_csv("Kmeans\data\players_22.csv")
model.load_data(data)
model.preprocess()
model.fit()