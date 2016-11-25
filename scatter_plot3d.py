import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

thedata_df = pd.read_csv('data/try-scatter-data.csv', header=0)
thedata = thedata_df[['murder', 'forcible_rape', 'robbery']].values
print thedata
# tampilkan row
print thedata[:,0]
print min(thedata[:,0])
print max(thedata[:,0])

# tampilkan column
# print thedata[0,:]
# data

def normalize(v, vmin, vmax):
    return (v - vmin)/(vmax - vmin)

"""


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5), ('g', 'o', -10, -45)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Scatter Plot')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

min_xs = min(thedata[:,0])
max_xs = max(thedata[:,0])
min_ys = min(thedata[:,1])
max_ys = max(thedata[:,1])
min_zs = min(thedata[:,2])
max_zs = max(thedata[:,2])

for color in ['r']:
    xs = normalize(thedata[:,0], min_xs, max_xs)
    ys = normalize(thedata[:,1], min_ys, max_ys)
    zs = normalize(thedata[:,2], min_zs, max_zs)
    print xs, ys, zs
    ax.scatter(xs, ys, zs, color=color, marker='o', s=30)
plt.show()
