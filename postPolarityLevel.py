# this is to cluster posts into 3 polarity score
import load_all_dataset as ld
import pandas as pd
import csv


# LOAD THE DATASET
# all dataset
all_df = ld.all_post()
post_cluster_df = ld.post_cluster_result()
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

# print all_df_clustered.dtypes
# print all_df_clustered.head(5)

postId = all_df_clustered['postId'].values
labels = all_df_clustered['3polarityLevel'].values

output_file = open("data/post_3polaritylevel.csv", "wb")
open_file_object = csv.writer(output_file)
open_file_object.writerow(["postId","resultCluster"])
open_file_object.writerows(zip(postId, labels))
output_file.close()