import load_all_dataset as ld
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def normalize(df):
    normal_df = (df[zone_columns] - df[zone_columns].min())/(df[zone_columns].max() - df[zone_columns].min())
    return normal_df

def scaler_modified(df):
    standard_scaler = StandardScaler()
    scaled_df = pd.DataFrame(standard_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    return scaled_df

def robust_modified(df):
    robust_scaler = RobustScaler()
    robust_df = pd.DataFrame(robust_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    return robust_df

def percentage(df):
    return (df - df.min())/(df.max() - df.min())

def smoothing():
    # piye carane
    pass

def correlated(df, ai):
    # correlation between Active Interests and all features
    topk = 7
    correlation = df[list(set(features)-set(zone_columns))+[ai]].corr()[ai].abs()
    # print 'correlation ', correlation
    n = zip(correlation.nlargest(topk).keys(), correlation.nlargest(topk))
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    return nlargest

# LOAD THE DATASET
# ds_file = 'data/english_foodgroup_new.json'
# ds_file = 'data/english_TEDtranslate_new.json'
# ds_file = 'data/english_traveladdiction_new.json'
ds_file = ['data/english_foodgroup_new.json', 'data/english_TEDtranslate_new.json', 'data/english_traveladdiction_new.json']

# profilephoto
# photo_file = 'data/album_english_foodgroups.json'
# photo_file = 'data/album_english_TEDtranslate.json'
# photo_file = 'data/album_english_traveladdiction.json'
photo_file = ['data/album_english_foodgroups.json', 'data/album_english_TEDtranslate.json', 'data/album_english_traveladdiction.json']
friendsnum_file = ['data/english_foodgroups_friendsnum.json', 'data/english_TEDtranslate_friendsnum.json', 'data/english_traveladdiction_friendsnum.json']

all_df = ld.new_dataset(ds_file)
all_df = all_df[['UserID', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity', 'ActiveInterests']].dropna()
# print all_df.dtypes
# quit()

# separate into zone
all_df['zone'] = 0
"""
# zone po
all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'po'
# zone ps
all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ps'
# zone nto
all_df['zone'].loc[ ((all_df['PostTextPolarity'] == 0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'nto'
# zone nts
all_df['zone'].loc[ ((all_df['PostTextPolarity'] == 0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'nts'
# zone ngo
all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'ngo'
# zone ngs
all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ngs'
"""
"""
# 0 as negative
# zone po
all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'po'
# zone ps
all_df['zone'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ps'
# zone ngo
all_df['zone'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'ngo'
# zone ngs
all_df['zone'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ngs'
"""

# 0 as positive
# zone po
all_df['zone'].loc[ ((all_df['PostTextPolarity'] >= 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'po'
# zone ps
all_df['zone'].loc[ ((all_df['PostTextPolarity'] >= 0) & (all_df['PostTextPolarity'] <= 1.0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ps'
# zone ngo
all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                    ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextSubjectivity'] <= 0.5)) ] = 'ngo'
# zone ngs
all_df['zone'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) &
                    ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextSubjectivity'] <= 1.0)) ] = 'ngs'

# all_df['Interactions'] = all_df['LikesCount']+all_df['SharesCount']+all_df['CommentsCount']
zone_dummies = pd.get_dummies(all_df['zone'])

all_df = all_df.join(zone_dummies)

# aggregation type
# dict_aggr_mean = {'UserId':np.min, 'po':np.mean, 'ps':np.mean, 'nto':np.mean, 'nts':np.mean, 'ngo':np.mean, 'ngs':np.mean}
# dict_aggr_sum = {'UserId':np.min, 'po':np.sum, 'ps':np.sum, 'nto':np.sum, 'nts':np.sum, 'ngo':np.sum, 'ngs':np.sum}
dict_aggr_mean = {'po':np.mean, 'ps':np.mean, 'ngo':np.mean, 'ngs':np.mean}
dict_aggr_sum = {'po':np.sum, 'ps':np.sum, 'ngo':np.sum, 'ngs':np.sum}

dict_feature = dict_aggr_mean.copy()
dict_feature.update({'ActiveInterests':np.min})
aggr_df = all_df.groupby(['UserID'], sort=True).agg(dict_feature).reset_index()

# aggr_df = aggr_df[['UserID', 'PostTextLength', 'LikesCount', 'SharesCount', 'CommentsCount', 'po', 'ps', 'ngo', 'ngs']]
# print 'non-aggregated df ', aggr_df

# features 'Followers'
behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','PostTextLengthMedian','SharedNewsSum','UploadVideoSum','UploadPhotoSum','Interaction', 'GenderCode']

log_behavior = ['LogFriends','LogProfilePhotos','LogCoverPhotos','LogPosts','LogPostTextLengthMedian', 'LogSharedNewsSum','LogUploadVideo','LogUploadPhoto','LogInteraction', 'GenderCode']

pi = ['Arts_and_Entertainment', 'Business_and_Services', 'Community_and_Organizations', 'Education_y', 'Health_and_Beauty', 'Home_Foods_and_Drinks', 'Law_Politic_and_Government',
      'News_and_Media', 'Pets_And_Animals', 'Recreation_and_Sports', 'Regional', 'Religion_and_Spirituality', 'Restaurants_and_Bars', 'Technology_Computers_and_Internet', 'Transportation_Travel_and_Tourism']


# obtain NoPosts, SharedNewsSum, UploadVideoSum
df_1 = ld.aggr_feature_user(ds_file)
aggr_df = pd.merge(aggr_df, df_1, how='inner', left_on='UserID', right_on='UserID')

# obtain about
df_2 = ld.about_dataset(ds_file)
aggr_df = pd.merge(aggr_df, df_2, how='inner', left_on='UserID', right_on='UserID')

# UserID, NoProfilePhotos, NoCoverPhotos, NoUploadedPhotos, NoPhotos
df_3 = ld.photo_dataset(photo_file)
aggr_df = pd.merge(df_3, aggr_df, how='inner', left_on='UserID', right_on='UserID')

# NumOfFriends
df_4 = ld.friendsnum_dataset(friendsnum_file)
aggr_df = pd.merge(df_4, aggr_df, how='inner', left_on='UserID', right_on='UserID')

# separate golden standard column name
target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
aggr_df = aggr_df.join(target_dummies)
target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

# columns = aggr_df.columns.tolist()
zone_columns = [ x for x in dict_aggr_mean ]

x_df = robust_modified(aggr_df[zone_columns])
aggr_df[zone_columns] = x_df

features = zone_columns + log_behavior #log_behavior #behavior #pi

"""
# normalize features
aggr_df = normalize(aggr_df[columns])
aggr_df = scaler_modified(aggr_df[columns])
"""

# multiple classifier
models = [LogisticRegression(), RandomForestClassifier(n_estimators=100), GaussianNB(), DecisionTreeClassifier(), SVC()]
names = ["Logistic Regression", "Random Forests Classifier", "Naive Bayes", "Decision Tree", "SVC"]
gabung = zip(models, names)

aggr_df['NoProfilePhotos'] = aggr_df['NoProfilePhotos'].astype(int)
aggr_df['NoCoverPhotos'] = aggr_df['NoCoverPhotos'].astype(int)

aggr_df['Interaction'] = np.log10(aggr_df['SharesCountSum']+aggr_df['CommentsCountSum']+aggr_df['LikesCountSum'])
aggr_df['LogFriends'] = np.log10(aggr_df['Friends'])
aggr_df['LogPosts'] = np.log10(aggr_df['NoPosts'])
aggr_df['LogProfilePhotos'] = np.log10(aggr_df['NoProfilePhotos'])
aggr_df['LogCoverPhotos'] = np.log10(aggr_df['NoCoverPhotos'])
aggr_df['LogPostTextLengthMedian'] = np.log10(aggr_df['PostTextLengthMedian'])
aggr_df['LogSharedNewsSum'] = np.log10(aggr_df['SharedNewsSum'])
aggr_df['LogUploadVideo'] = np.log10(aggr_df['UploadVideoSum'])
aggr_df['LogUploadPhoto'] = np.log10(aggr_df['UploadPhotoSum'])
aggr_df['LogInteraction'] = np.log10(aggr_df['Interaction'])

# logfriends, loguploadvideo
aggr_df = aggr_df.replace(-np.inf, 0)

# print aggr_df.head()
# aggr_df.to_csv('test_aggr_df.csv', sep=',', encoding='utf-8')
# quit()

numCV = 3
print numCV, '-fold CV'
print target
print zone_columns
for model, name in gabung:
    print name
    print 'MEAN ACCURACY \t MACRO F1 \t TARGET \t FEATURES'
    for i, ai in enumerate(target):
        if any('Friends' in x for x in features):
            corr = correlated(aggr_df, ai)
            arr_corr = [ x for (x, y) in corr ]
            value_corr = [ y for (x, y) in corr ]
            cleaned_features = zone_columns + arr_corr
            # print 'cleaned features ', cleaned_features
        else:
            cleaned_features = features

        # print i, v
        # print aggr_df[features]
        # print aggr_df[target][v].astype(int)
        # print aggr_df[cleaned_features]
        #"""
        cv_model_accuracy = cross_val_score(model, aggr_df[cleaned_features], aggr_df[target][ai], cv=numCV, scoring='accuracy')
        cv_model_f1_macro = cross_val_score(model, aggr_df[cleaned_features], aggr_df[target][ai], cv=numCV, scoring='f1_macro')
        print cv_model_accuracy.mean(), '\t', cv_model_f1_macro.mean(), '\t', ai, '\t', cleaned_features, '\t', value_corr
        #"""
    print '\n'