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
feature_user = ld.averageSum_Feature_User()
user_cluster_result = ld.user_cluster_result('data/post_3polaritylevel.csv')

print feature_user.head(5)
print feature_user.dtypes
print user_cluster_result.info()
print user_cluster_result.head(5)

quit()

# merge with post clustered result
all_df_clustered = user_cluster_result.join(feature_user, on='userId', how='left', lsuffix='_left')
all_df_clustered['resultCluster'] =all_df_clustered['resultCluster'].astype('category')
print all_df_clustered.head(10)
print all_df_clustered.dtypes

quit()
# CORRELATION BETWEEN VARIABLES
# resultCluster dan LikesCount
# Calculate the correlation coefficient
r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['LikesCount'])
print ('point biserial correlation r is %s with p = %s' %(r,p))
"""
fig, ax = plt.subplots(figsize=(12,4))
sns.boxplot(y = 'resultCluster', x = 'LikesCount', data = all_df_clustered, width = 0.8,orient = 'h', showmeans = True, fliersize = 3, ax = ax)
plt.show()
"""

# resultCluster dan SharesCount
r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['SharesCount'])
print ('point biserial correlation r is %s with p = %s' %(r,p))

r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['CommentsCount'])
print ('point biserial correlation r is %s with p = %s' %(r,p))

r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['PostTextLength'])
print ('point biserial correlation r is %s with p = %s' %(r,p))

r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['SharedNews'])
print ('point biserial correlation r is %s with p = %s' %(r,p))

r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['UploadPhoto'])
print ('point biserial correlation r is %s with p = %s' %(r,p))

r, p = stats.pointbiserialr(all_df_clustered['resultCluster'], all_df_clustered['SharesCount'])
print ('point biserial correlation r is %s with p = %s' %(r,p))

# spearmans
r, p = stats.spearmanr(all_df_clustered['LikesCount'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and LikesCount is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['SharesCount'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and SharesCount is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['CommentsCount'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and CommentsCount is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['PostTextLength'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and PostTextLength is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['SharedNews'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and SharedNews is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['UploadPhoto'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and UploadPhoto is %s with p = %s' %(r,p))

r, p = stats.spearmanr(all_df_clustered['SharesCount'], all_df_clustered['resultCluster'])
print ('spearman correlation r between resultCluster and SharesCount is %s with p = %s' %(r,p))