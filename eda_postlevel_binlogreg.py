import load_all_dataset as ld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# LOAD THE DATASET
# all dataset
all_df = ld.all_post()
file_post_cluster = 'D:/githubrepository/text-analysis-master/09.01.2016/kmeans_post_result_09012016.csv'
post_cluster_df = ld.post_cluster_result(file_post_cluster)
# merge with post clustered result
all_df_clustered = pd.merge(all_df, post_cluster_df, on='postId', how='inner')
all_df_clustered = all_df_clustered[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextSubjectivity', 'PostTextPolarity', 'resultCluster']]
all_df_clustered['Interactions'] = all_df_clustered['LikesCount']+all_df_clustered['SharesCount']+all_df_clustered['CommentsCount']
all_df_clustered['resultCluster'] =all_df_clustered['resultCluster'].astype('category',ordered=False)

# bikin level polarity into 3
all_df_clustered['3polarityLevel'] = 0
all_df_clustered['3polarityLevel'].loc[all_df_clustered['PostTextPolarity'] > 0] = 1
all_df_clustered['3polarityLevel'].loc[all_df_clustered['PostTextPolarity'] < 0] = -1
all_df_clustered['3polarityLevel'] =all_df_clustered['3polarityLevel'].astype('category')

all_df_clustered['binLike'] = 0
all_df_clustered['binLike'].loc[all_df_clustered['LikesCount'] > 0] = 1
all_df_clustered['binLike'] = all_df_clustered['binLike'].astype('category',ordered=False)

all_df_clustered['binComment'] = 0
all_df_clustered['binComment'].loc[all_df_clustered['CommentsCount'] > 0] = 1
all_df_clustered['binComment'] = all_df_clustered['binComment'].astype('category',ordered=False)

all_df_clustered['binInt'] = 0
all_df_clustered['binInt'].loc[all_df_clustered['Interactions'] > 0] = 1
all_df_clustered['binInt'] = all_df_clustered['binInt'].astype('category',ordered=False)

# bikin level polarity into 3
all_df_clustered['3polarityLevel'] = 0
all_df_clustered['3polarityLevel'].loc[all_df_clustered['PostTextPolarity'] > 0] = 1
all_df_clustered['3polarityLevel'].loc[all_df_clustered['PostTextPolarity'] < 0] = -1
all_df_clustered['3polarityLevel'] =all_df_clustered['3polarityLevel'].astype('category')

# bikin level subjectivity into 2
all_df_clustered['2subjLevel'] = 0
all_df_clustered['2subjLevel'].loc[all_df_clustered['PostTextSubjectivity'] >= 0.5] = 1

print all_df_clustered.head(5)

# print all_df_clustered[['PostTextPolarity', 'PostTextSubjectivity', 'PostTextLength']]
# print all_df_clustered['binLike']

# binlogreg, we want to predict the binLike column using PostTextPolarity, PostTextSubjectivity, PostTextLength
# print all_df_clustered.head(10)
# print pd.crosstab(all_df_clustered['binLike'], all_df_clustered['3polarityLevel'])
logit = sm.Logit(all_df_clustered['binInt'], all_df_clustered[['3polarityLevel', '2subjLevel', 'PostTextLength']])
result = logit.fit()
print result.summary()

# odds ratios
print ("Odds Ratios")
print (np.exp(result.params))

# odd ratios with 95% confidence intervals
print ('Odd ratios with 95% confidence intervals')
conf = result.conf_int()
print conf
