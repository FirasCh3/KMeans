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
class KMeans:
    def __init__(self, k=3, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.data=None
    def preprocess(self):
        self.data.dropna(inplace=True)
    def load_data(self, data):
        cols = ["overall", "value_eur"]
        data = data[cols]
        data = data.loc[:1000, :]
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data.to_numpy()))
        data.columns = cols
        self.data = data
        self.centroids = (data.sample(n=self.k))
        self.centroids.index = [i for i in range(self.k)]
    def fit(self):         
        distance = pd.DataFrame(columns=[i for i in range(self.k)], index = [i for i in self.data.index])
        for i in self.data.index:
            for j in range(self.k):
                distance.loc[i, j] = np.sqrt((self.data.loc[i, :] - self.centroids.loc[j, :])**2).sum()
        labels = distance.idxmin(axis=1)
        
       




model = KMeans()
data = pd.read_csv("Kmeans\data\players_22.csv", low_memory=False)
model.load_data(data)
model.preprocess()
model.fit()