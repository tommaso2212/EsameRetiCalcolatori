import pandas as pd
from K_meansCUDA import K_meansCUDA
import matplotlib.pyplot as plt
import time

start_time = time.time()

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 63, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 23, 19, 7, 24]
})

k_means = K_meansCUDA(df, 3)
df, centroids = k_means.train()

print("--- %s seconds ---" % (time.time() - start_time))

color = ['b', 'g', 'r']
fig = plt.figure(figsize=(5, 5))
for i in range(df.shape[0]):
    plt.scatter(df['x'][i], df['y'][i], color=color[int(df['cluster'][i])], alpha=0.5, edgecolor='k')

plt.scatter(centroids[0][0], centroids[0][1], color='b', marker="x")
plt.scatter(centroids[1][0], centroids[1][1], color='g', marker="x")
plt.scatter(centroids[2][0], centroids[2][1], color='r', marker="x")
plt.show()