import pandas as pd
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# K-means implemented with parallel algorithm
class K_meansCUDA:
    # Constructor
    def __init__(self, dataframe, num_cluster): # requires dataset in the format of pandas dataframe and cluster's number as arguments
        self.df = dataframe

        self.features_gpu = gpuarray.to_gpu(self.df.to_numpy().astype(np.float32)) # allocates GPU memory in which stores feature values

        self.final_df = dataframe.copy(deep=False) # makes a copy of the original dataset
        self.final_df['cluster'] = np.zeros(self.final_df.shape[0], np.int32) # adds a column in which store labels
        self.k = num_cluster
        self.centroids = np.array(self.df.iloc[:self.k]).astype(np.float32) # sets initial centroids as first k elements of the dataset
        self.init_kernels()

    def init_kernels(self):
        mod = SourceModule("""
            __global__ void assignment(int *labels, float *features, float *centroids, int *k, int *features_height, int *features_width){
                extern __shared__ float distances[];   //creates a temporal array

                int idx = blockIdx.x * blockDim.x + threadIdx.x;

                for(int i=0; i<*k; i++){    //for each cluster
                    float distance_eucl = 0;
                    for(int j=0; j<*features_width; j++){   //for each feature
                        distance_eucl += pow(features[idx + *features_height * j] - centroids[i + *k * j], 2);    //sums of the squares of the components
                    }
                    distance_eucl = sqrt(distance_eucl);    //square root of the sum
                    if(i==0){
                        distances[idx] = distance_eucl; //stores distance value 
                    }
                    if(distances[idx] > distance_eucl){ //if the distance stored is less than the calculated distance
                        distances[idx] = distance_eucl; //updates distance stored
                        labels[idx] = i;    //updates label
                    }
                }
                __syncthreads();
            }
        """)

        self.kernel_assign = mod.get_function("assignment") # references cuda kernel for the assignment phase

    def assignment(self):
        
        clusters_gpu = gpuarray.to_gpu(np.zeros(self.df.shape[0]).astype(np.int32)) # allocates GPU memory in which stores label values
        centroids_gpu = gpuarray.to_gpu(self.centroids.astype(np.float32)) # allocates GPU memory in which stores centroid values
    
        self.kernel_assign(clusters_gpu, self.features_gpu, centroids_gpu, cuda.In(np.int32(self.k)), cuda.In(np.int32(self.df.shape[0])), cuda.In(np.int32(self.df.shape[1])), block=(self.df.shape[0],1,1), shared=8*self.df.shape[0]) # executes assignment kernel
        
        self.final_df['cluster'] = clusters_gpu.get() # get calculated labels

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
