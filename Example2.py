import pandas as pd
from K_means import K_means
from K_meansCUDA import K_meansCUDA
import matplotlib.pyplot as plt
import time

# Defines dataset as pandas dataframe
df = pd.read_csv("a1.csv")
df = df.dropna()/10000
df = df.iloc[:1000]

# Defines cluster number
num_cluster = 5

# Initializes procedural K_means
k_means = K_means(df, num_cluster)

# Initializes parallel K_means
k_meansCUDA = K_meansCUDA(df, num_cluster)

# Execute procedural K_means
start_time = time.time()
procedural_df, procedural_centroids = k_means.train()
procedural_time = time.time() - start_time

print("Procedural execution time: " + str(procedural_time))

# Execute parallel K_means
start_time = time.time()
parallel_df, parallel_centroids = k_meansCUDA.train()
parallel_time = time.time() - start_time

print("Parallel execution time: " + str(parallel_time))

# Plots results
color = ['b', 'g', 'r', 'c', 'm']

fig_procedural = plt.figure("Procedural", figsize=(5, 5))
for i in range(df.shape[0]):
    plt.scatter(procedural_df['x'][i], procedural_df['y'][i], color=color[int(procedural_df['cluster'][i])], alpha=0.5, edgecolor='k')
for i in range(num_cluster):
    plt.scatter(procedural_centroids[i][0], procedural_centroids[i][1], color="k", marker="x")
plt.title("execution time: " + str(procedural_time))
plt.suptitle("Procedural algorithm")
plt.xlabel("x")
plt.ylabel("y")


fig_parallel = plt.figure("Parallel", figsize=(5, 5))
for i in range(df.shape[0]):
    plt.scatter(parallel_df['x'][i], parallel_df['y'][i], color=color[int(parallel_df['cluster'][i])], alpha=0.5, edgecolor='k')
for i in range(num_cluster):
    plt.scatter(parallel_centroids[i][0], parallel_centroids[i][1], color="k", marker="x")
plt.title("execution time: " + str(parallel_time))
plt.suptitle("Parallel algorithm")
plt.xlabel("x")
plt.ylabel("y")
plt.show()