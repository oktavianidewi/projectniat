import json
import pandas as pd
from collections import Counter
import csv
import numpy as np
import load_all_dataset as ld

# dari basilisa UserNum -> 1 - 341, UserID -> adamdolhanyk,
# gabung dengan num of shared news di UserID
#
def openfile(filenamearrs):
    if type(filenamearrs) is list:
        # print 'list'
        data = {}
        for filename in filenamearrs:
            with open(filename) as file:
                dict = json.load(file)
            data.update(dict)
    else:
        # print 'bukan list'
        with open(filenamearrs) as file:
            data = json.load(file)
    return data

def readcsvfile(presentFile):
    # content
    # presentFile = 'data/english_constitutionalpatriot 2.json'
    data_file = open(presentFile)
    json_data = json.load(data_file)
    return json_data

def about_extractor(file):
    json_data = openfile(file)
    z = {}
    y = []
    for username in json_data:
        if 'about' in json_data[username]:
            z = json_data[username]["about"]
            z['UserID'] = username
            y.append(z)
    df = pd.DataFrame(y)
    # print df.head()
    # print df.info()
    df['UserID'] = df['UserID'].astype(str)
    return df

def grouped_category(filename):
    json_data = openfile(filename)

    # baca file csv dmoz2all
    dict_zgroup = {}
    filename_newcat = 'data/dmoz2all.csv'
    outfile_newcat = file(filename_newcat,'rb')
    read_csvfile_newcat = csv.reader(outfile_newcat, delimiter=',', quotechar='|')
    zgroup = [ items for items in read_csvfile_newcat ]

    # print zgroup
    for index in zgroup:
        # print index[0]
        dict_zgroup[index[0]] = index[2]
    print dict_zgroup
    # print len(dict_zgroup)

    z = {}
    p = {}
    q = {}
    x = []
    y = {}
    for username in json_data:
        #z['username'] = str(username)
        #y.append(z)
        if "like" in json_data[username]:
            z = json_data[username]["like"]
            y[username] = {x.lower():z.count(x) for x in z}
    group_besar = {}
    big_group_arr = []
    print x
    for username in y:
        x_per_user = y[username]
        z = []
        for x_per_user_like in x_per_user:
            if x_per_user_like in dict_zgroup:
                big_group = dict_zgroup[x_per_user_like]
            else:
                big_group = 'Other'
            # print '  ', username ,' ', x_per_user_like, ' jumlah ', x_per_user[x_per_user_like] ,' grup ', big_group
            big_group_arr.append([username, x_per_user_like, x_per_user[x_per_user_like], big_group])
    # print big_group_arr
    df = pd.DataFrame(big_group_arr, columns=['UserID', 'category', 'num_of_category', 'group_of_category'])
    # print df.head(5)
    df['UserID'] = df['UserID'].astype(str)
    sum_df = df.groupby(['UserID', 'group_of_category'], sort=False).agg({
            'num_of_category':np.sum
        })
    return sum_df.reset_index()

"""
presentFile = ['data/english_constitutionalpatriot 2.json',
               'data/english_Gu.json',
               'data/english_Hiking With Dogs 2.json',
               'data/english_Jazzmasters&Jaguars 2.json',
               'data/english_Like For Like Promote Your Business 2.json']
presentFile = 'data/english_constitutionalpatriot 2.json'
"""
presentFile = 'data/english_traveladdiction.json'

about_df = about_extractor(presentFile)
grouped_category_df = grouped_category(presentFile)

pivot_grouped_category_df = grouped_category_df.pivot(index='UserID', columns='group_of_category', values='num_of_category').reset_index()
# print pivot_grouped_category_df.head()
merge_about_likedcat = about_df.merge(pivot_grouped_category_df, on='UserID', how='left')
merge_about_likedcat['UserIDs'] = merge_about_likedcat['Facebook'].str.split('/').str.get(1)
merge_about_likedcat['UserID'] = merge_about_likedcat['UserID'].astype('str')

# print merge_about_likedcat
merge_about_likedcat.to_csv('test_merge_about_likedcat.csv', sep=',', encoding='utf-8')
quit()

# bagaimana gabungin data dari basilisa dan crawling pada kolom UserID?
# manually combine dari  file merge_about_likedcat.csv dengan user_features_all.csv

# menggabungkan about dengan all fitur dari basilisa
user_feature = ld.averageSum_Feature_User()

concat_user_feature_about = user_feature.merge(merge_about_likedcat, left_on='UserID', right_on = 'UserIDs', how='left')
# concat_user_feature_about = pd.concat([user_feature, merge_about_likedcat], axis=1)
# concat_user_feature_about_category = concat_user_feature_about.join(pivot_grouped_category_df, on='UserID', how='left', lsuffix='_')
print concat_user_feature_about.head()
print concat_user_feature_about.tail()
concat_user_feature_about.to_csv('user.csv', sep=',', encoding='utf-8')
quit()