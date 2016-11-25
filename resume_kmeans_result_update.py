import csv
import collections
import numpy as np
import pandas as pd
from pandas import stats

"""
source = pd.DataFrame({'Country' : ['USA', 'USA', 'Russia','USA'],
                  'City' : ['New-York', 'New-York', 'Sankt-Petersburg', 'New-York'],
                  'Short name' : ['NY','New','Spb','NY']})

print source.groupby(['Country','City']).agg(lambda x:x.value_counts().index[0])

quit()

df_logfile = pd.DataFrame({
    'host' : ['this.com', 'this.com', 'this.com', 'that.com', 'other.net',
              'other.net', 'other.net'],
    'service' : ['mail', 'mail', 'web', 'mail', 'mail', 'web', 'web' ] })

df = df_logfile.groupby(['host','service'])['service'].agg({'no':'count'})
mask = df.groupby(level=0).agg('idxmax')
df_count = df.loc[mask['no']]
df_count = df_count.reset_index()
print("\nOutput\n{}".format(df_count))

quit()

"""


csvfilename = 'post_3polaritylevel.csv'
kmeans_df = pd.read_csv('data/'+csvfilename, header=0)
kmeans_df['userId'] = kmeans_df['postId'].str.split('_').str.get(0).astype(int)
kmeans_df['postNum'] = kmeans_df['postId'].str.split('_').str.get(1).astype(int)
kmeans_df = kmeans_df.drop(['postId', 'postNum'], axis=1)

userId = kmeans_df['userId']
userId = userId.drop_duplicates()

usercluster = kmeans_df.groupby(['userId']).agg(lambda x:x.value_counts().index[0])
userId_np = userId.values
usercluster_np = usercluster['resultCluster'].values

resultUserCluster_file = open("data/resume_"+csvfilename, "wb")
open_file_object = csv.writer(resultUserCluster_file)
open_file_object.writerow(["userId","resultCluster"])
open_file_object.writerows(zip(userId_np, usercluster_np))
resultUserCluster_file.close()