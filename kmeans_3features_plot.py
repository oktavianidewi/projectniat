import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
import pandas as pd
import json
import sys
import load_all_dataset as ld
from sklearn.metrics import silhouette_score
import math

# plt.scatter(x,y)
# plt.show()

k = 4
# data = np.mat(ownKmeans.loadDataSet('08.22.2016/271_all_post_processed_humanjudge7_subjpol - withoutNullValueSubjPol - onlyNumbers#2.txt'))

df = ld.all_post()
print df.dtypes
df['userId'] = df['postId'].str.split('_').str.get(0).astype(int)
df['postIds'] = df['postId'].str.split('_').str.get(1).astype(int)

df['interaction'] = df['LikesCount']+df['SharesCount']+df['CommentsCount']
df = df[['userId', 'postIds', 'postId', 'PostTextLength', 'interaction', 'PostTextSubjectivity', 'PostTextPolarity']].dropna()

userId = df['userId'].values
postId = df['postIds'].values
polarity = df['PostTextPolarity'].values
subjectivity = df['PostTextSubjectivity'].values

# data = df[['PostTextLength', 'interaction', 'PostTextSubjectivity', 'PostTextPolarity']].values
data = df[['PostTextSubjectivity', 'PostTextPolarity', 'PostTextLength']].values

# gabung comments, likes, shares

"""
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]
"""
# data = np.array([[1, 2], [1.5, 6], [2, 3] , [5, 6], [6, 6], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6],[9, 11]])

kmeans = KMeans(n_clusters=k)
kmeans.fit(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print centroids
print postId
print labels

"""
resultcluster_file = open("kmeans5_result_10112016.csv", "wb")
open_file_object = csv.writer(resultcluster_file)
open_file_object.writerow(["userId","postId","subjectivity","polarity","resultCluster"])
open_file_object.writerows(zip(userId, postId, subjectivity, polarity, labels))
resultcluster_file.close()
"""

"""
# untuk membuat cluster dalam kolom sendiri2
file_csv = 'kmeans_result_09172016.csv'
df = pd.read_csv(file_csv, header=0)
df['UserNum'] = df['postId'].str.split('_').str.get(0).astype(int)

# resultcluster level
resultCluster_dummies = pd.get_dummies(df['resultCluster'])
df = df.join(resultCluster_dummies)
df.rename(columns={0: 'Cluster_0', 1: 'Cluster_1', 2: 'Cluster_2', 3: 'Cluster_3'}, inplace=True)
print df.dtypes
df.to_csv('kmeans_result_09172016.csv')


# np.savetxt('08.22.2016/271_all_post_processed_humanjudge7_subjpol - withoutNullValueSubjPol - onlyNumbers_resultKmeansScikit.txt', labels, delimiter='\t')
# kalo bisa di pas2in warna green -> sharer, red -> monologue, turquoise -> sociable, blue -> critics
title = 'Scatter Plot Distribution'
opsiwarna = ['turquoise', 'red', 'green', 'blue', 'black']

for i in range(k):
    # select only data observations with cluster label == i
    # tentukan index mana aja yang di-print disini
    # ds = data[262:297]
    ds = data[np.where(labels==i)]
    # print 'ds awal ', ds
    # print 'ds ', ds[0:2]
    # plot the data observations
    # plt.plot(x,y,'o')
    # print 'ds ', ds
    plt.plot(ds[:,0],ds[:,1], 'o', color=opsiwarna[i])
    # plt.plot(ds[:,0],ds[:,1], 'o', color='white')
    # plot the centroids
    lines = plt.plot(centroids[i,0], centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines, ms=10.0)
    plt.setp(lines, mew=2.0)
plt.title(title)
plt.xlabel('PostTextSubjectivity')
plt.ylabel('PostTextPolarity')
plt.axis([-0.2, 1.2, -1.5, 1.5])
plt.show()
"""