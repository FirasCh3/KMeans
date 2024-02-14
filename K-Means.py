"""
Algorithm description:
    for each cluster create a random centroid for each features
    example: for k = 3, cluster  will have 3 values for overall features etc...
    calculate distance between each data point and clusters then assign its label the closest cluster
    finally update the clusters to the mean of all the points in it
"""