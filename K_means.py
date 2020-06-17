import pandas as pd
import numpy as np

# K-means implemented with procedural algorithm
class K_means:
    # Constructor
    def __init__(self, dataframe, num_cluster): # requires dataset in the format of pandas dataframe and cluster's number as arguments
        self.df = dataframe
        self.final_df = dataframe.copy(deep=False) # makes a copy of the original dataset
        self.final_df['cluster'] = np.zeros(self.final_df.shape[0], np.int32)   # adds a column in which store labels
        self.k = num_cluster
        self.centroids = np.array(self.df.iloc[:self.k]).astype(np.float32) # sets initial centroids as first k elements of the dataset

    # This method assigns dataset points to a cluster
    def assignment(self):
        distance_eucl = np.zeros(self.k) # creates a temporal numpy array 
        for index, row in self.df.iterrows(): # for each row of the dataframe
            for j in range(self.k): # for each cluster
                distance_eucl[j] = np.linalg.norm(row-self.centroids[j]) # calculates euclidean distance between dataset's point and centroid and stores it in the temporal array
            self.final_df.loc[index,'cluster'] = np.argmin(distance_eucl) # assigns the point to the nearest centroid
    
    # This method updates centroids, return True if reaches the optimum centroids
    def update_centroids(self):
        flag_optimum = False # boolean flag for optimum
        for i in range(self.k): # for each cluster
            mean = np.mean(self.final_df[self.final_df['cluster'] == i]).astype(np.float32) # calculates the average between points that are assigned to the same cluster
            for j in range(self.df.shape[1]):  # for each features
                if(mean[j] == self.centroids[i][j]):    # True if the centroids has no need to be updated
                    flag_optimum = True
                else:
                    flag_optimum = False
                    self.centroids[i][j] = mean[j]  # Updates centroid component values
        return flag_optimum

    # This method trains the network using a loop for alternate the assignment phase and the updating phase until it reaches the optimum.
    def train(self):
        while True:
            self.assignment()
            flag = self.update_centroids()
            if(flag):
                break
        return self.final_df, self.centroids    # When the optimum value is reached, it returns the labeled dataframe and the centroids found


