import sklearn.datasets as datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x, y = datasets.make_moons(n_samples = 1000, shuffle = False, noise = 0.2)

df = pd.DataFrame({
    "x1" : x[:,0],
    "x2" : x[:,1],
    "y"  : y
})

plt.figure(figsize=(16,12))
ax = plt.axes(projection='3d')
ax.scatter(df.x1, df.x2, df.x1 * df.x2, c = df.y, cmap = "viridis")