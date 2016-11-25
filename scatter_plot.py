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

# data = np.mat(ownKmeans.loadDataSet('08.22.2016/271_all_post_processed_humanjudge7_subjpol - withoutNullValueSubjPol - onlyNumbers#2.txt'))

df = ld.all_post()
df['interaction'] = df['LikesCount']+df['SharesCount']+df['CommentsCount']
df = df[['postId', 'UserName', 'PostTextLength', 'interaction', 'PostTextSubjectivity', 'PostTextPolarity']].dropna()


k = [-1, 0, 1]
df['polarityLevel'] = 0
df['polarityLevel'].loc[df['PostTextPolarity'] > 0] = 1
df['polarityLevel'].loc[df['PostTextPolarity'] < 0] = -1
# df['polarityLevel'] = df['polarityLevel'].astype('category')

k_subj = [0, 1]
df['subjectivityLevel'] = 0
df['subjectivityLevel'].loc[df['PostTextSubjectivity'] > 0] = 1


"""
postId = df['postId'].values
polarity = df['PostTextPolarity'].values
subjectivity = df['PostTextSubjectivity'].values
"""
# data = df[['PostTextLength', 'interaction', 'PostTextSubjectivity', 'PostTextPolarity']].values

x = df[['PostTextSubjectivity', 'PostTextPolarity', 'polarityLevel', 'subjectivityLevel']][df['UserName'] == 'Dawn Deangelo']
data = x.values

# np.savetxt('08.22.2016/271_all_post_processed_humanjudge7_subjpol - withoutNullValueSubjPol - onlyNumbers_resultKmeansScikit.txt', labels, delimiter='\t')
# kalo bisa di pas2in warna green -> sharer, red -> monologue, turquoise -> sociable, blue -> critics
title = 'Scatter Plot Distribution (Polarity Level)'
opsiwarna = ['turquoise', 'red', 'black']
opsiwarna_subj = ['green', 'purple']

"""
Sociable User: Diana Shay Diehl, index = [11819:11839]
Sharer User: Gaylie Blake, index = [14178:14229]
Monologue User: Adam Soergel, index = [262:297]

"""
# for i in range(k):
for i in k:
    # select only data observations with cluster label == i
    # tentukan index mana aja yang di-print disini
    # ds = data[262:297]
    # 3 -> subj, 2-> obj
    ds = data[np.where( data[:,2] == i)]
    # print 'ds awal ', ds
    # print 'ds ', ds[0:2]
    # plot the data observations
    # plt.plot(x,y,'o')
    # print 'ds ', ds
    plt.plot(ds[:,0],ds[:,1], 'o', color=opsiwarna[i])
    # plt.plot(ds[:,0],ds[:,1], 'o', color='white')
    # plot the centroids
    # lines = plt.plot(centroids[i,0], centroids[i,1],'kx')
    # make the centroid x's bigger
    # plt.setp(lines, ms=10.0)
    # plt.setp(lines, mew=2.0)
plt.title(title)
plt.xlabel('PostTextSubjectivity')
plt.ylabel('PostTextPolarity')
plt.axis([-0.2, 1.2, -1.5, 1.5])
plt.show()