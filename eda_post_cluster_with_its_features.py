# this is EDA (Explanatory Data Analaysis between posts cluster and its related features)
import load_all_dataset as ld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats


# LOAD THE DATASET
# all dataset
all_df = ld.all_post()
file_post_cluster = 'D:/githubrepository/text-analysis-master/09.01.2016/kmeans_post_result_09012016.csv'
post_cluster_df = ld.post_cluster_result(file_post_cluster)
# merge with post clustered result
all_df_clustered = pd.merge(all_df, post_cluster_df, on='postId', how='inner')
all_df_clustered = all_df_clustered[['postId', 'PostTime', 'PostTextPolarity', 'WeekendOrWeekday', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextSubjectivity', 'resultCluster']]
all_df_clustered['Interactions'] = all_df_clustered['LikesCount']+all_df_clustered['SharesCount']+all_df_clustered['CommentsCount']
all_df_clustered['InteractionsLC'] = all_df_clustered['LikesCount']+all_df_clustered['CommentsCount']
all_df_clustered['resultCluster'] =all_df_clustered['resultCluster'].astype('category',ordered=False)


"""
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearsonr = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

g = sns.PairGrid(all_df_clustered, vars = ['Interactions', 'PostTextSubjectivity', 'PostTextPolarity'], size = 3.5) # define the pairgrid
g.map_upper(plt.scatter)
g.map_diag(sns.distplot)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)
plt.show()
"""

"""
sns.jointplot(x="PostTextPolarity", y="Interactions", data=all_df_clustered, kind = 'reg', size = 5)
sns.jointplot(x="PostTextSubjectivity", y="SharesCount", data=all_df_clustered, kind = 'reg', size = 5)
sns.jointplot(x="PostTextSubjectivity", y="CommentsCount", data=all_df_clustered, kind = 'reg', size = 5)
sns.jointplot(x="PostTextSubjectivity", y="PostTextLength", data=all_df_clustered, kind = 'reg', size = 5)
"""

# bikin level polarity into 3
all_df_clustered['3polarityLevel'] = 0
all_df_clustered['3polarityLevel'].loc[all_df_clustered['PostTextPolarity'] > 0] = 1
all_df_clustered['3polarityLevel'].loc[all_df_clustered['PostTextPolarity'] < 0] = -1
all_df_clustered['3polarityLevel'] = all_df_clustered['3polarityLevel'].astype('category')

# all_df_clustered['binWeekend'] = all_df_clustered['binWeekend'].astype('category')
print all_df_clustered.head(30)
quit()
r, p = stats.spearmanr(all_df_clustered['BinWeekend'], all_df_clustered['Interactions'])
print ('spearman correlation r between BinWeekend and Interactions is %s with p = %s' %(r,p))
fig, ax = plt.subplots(figsize=(12,4))
sns.boxplot(y = 'BinWeekend', x = 'Interactions', data = all_df_clustered,width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
plt.show()
"""
# bikin level subjectivity into 2
all_df_clustered['2subjLevel'] = 0
all_df_clustered['2subjLevel'].loc[all_df_clustered['PostTextSubjectivity'] >= 0.5] = 1

# bikin level polarity into 5
all_df_clustered['5polarityLevel'] = 0
all_df_clustered['5polarityLevel'].loc[all_df_clustered['PostTextPolarity'] < 0] = -1
all_df_clustered['5polarityLevel'].loc[(all_df_clustered['PostTextPolarity'] > 0) & (all_df_clustered['PostTextPolarity'] <= 0.481)] = 1
all_df_clustered['5polarityLevel'].loc[(all_df_clustered['PostTextPolarity'] > 0.481) & (all_df_clustered['PostTextPolarity'] <= 0.850)] = 2
all_df_clustered['5polarityLevel'].loc[(all_df_clustered['PostTextPolarity'] > 0.850) & (all_df_clustered['PostTextPolarity'] <= 1.0)] = 3
all_df_clustered['5polarityLevel'] =all_df_clustered['5polarityLevel'].astype('category')


print all_df_clustered.head(10)
print all_df_clustered.dtypes
print all_df_clustered.describe()
"""

# histogram
# all_df_clustered['3polarityLevel'].hist(bins=50)
# temp = pd.crosstab(all_df_clustered['resultCluster'], all_df_clustered['Interactions'])
# temp.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# CORRELATION BETWEEN VARIABLES
# resultCluster dan LikesCount
# Calculate the correlation coefficient

# Calculate the correlation coefficient
# spearmans
"""
r, p = stats.spearmanr(all_df_clustered['3polarityLevel'], all_df_clustered['Interactions'])
print ('spearman correlation r between 3polarityLevel and Interactions is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['InteractionsLC'], all_df_clustered['3polarityLevel'])
print ('spearman correlation r between 3polarityLevel and InteractionsLC is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['LikesCount'], all_df_clustered['3polarityLevel'])
print ('spearman correlation r between 3polarityLevel and LikesCount is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['SharesCount'], all_df_clustered['3polarityLevel'])
print ('spearman correlation r between 3polarityLevel and SharesCount is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['CommentsCount'], all_df_clustered['3polarityLevel'])
print ('spearman correlation r between 3polarityLevel and CommentsCount is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['PostTextLength'], all_df_clustered['3polarityLevel'])
print ('spearman correlation r between 3polarityLevel and PostTextLength is %s with p = %s' %(r,p))


r, p = stats.spearmanr(all_df_clustered['PostTextSubjectivity'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and PostTextSubjectivity is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['PostTextPolarity'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and PostTextPolarity is %s with p = %s' %(r,p))

# fig, ax = plt.subplots(5, figsize=(15,4))
fig, ax = plt.subplots(figsize=(12,4))
sns.boxplot(y = '3polarityLevel', x = 'Interactions', data = all_df_clustered,width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
# sns.boxplot(y = 'resultCluster', x = 'SharesCount', data = all_df_clustered,width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
# sns.boxplot(y = 'resultCluster', x = 'CommentsCount', data = all_df_clustered,width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
# sns.boxplot(y = 'resultCluster', x = 'PostTextLength', data = all_df_clustered,width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
# sns.boxplot(y = 'resultCluster', x = 'Interactions', data = all_df_clustered,width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
plt.show()
"""