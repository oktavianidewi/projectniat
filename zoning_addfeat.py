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
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    normal_df = (df[zone_columns] - df[zone_columns].min())/(df[zone_columns].max() - df[zone_columns].min())
    return normal_df

def scaler_modified(df):
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    standard_scaler = StandardScaler()
    scaled_df = pd.DataFrame(standard_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    new_column = [x+'_scaled' for x in zone_columns]
    df[new_column] = scaled_df
    return df, new_column

def robust_modified(df):
    robust_scaler = RobustScaler()
    robust_df = pd.DataFrame(robust_scaler.fit_transform(df[zone_columns]), columns=[zone_columns])
    new_column = [x+'_robust' for x in zone_columns]
    df[new_column] = robust_df
    return df, new_column

def log_modified(df):
    # log function
    for column_name in zone_columns:
        print column_name
        df[column_name+'_log'] = np.log10(df[column_name])
    new_column = [x+'_log' for x in zone_columns]
    df = df.replace(-np.inf, 0)
    return df, new_column
"""
def percentage_modified(df):
    total = 0
    for i in zone_columns:
        total += df[i]
    df['total_post'] = total
    for i in zone_columns:
        df[i+'_percent'] = df[i]/df['total_post']
    new_column = [x+'_percent' for x in zone_columns]
    return df, new_column
"""
def smoothing():
    # piye carane
    pass

# task 1 remove the highly correlated pi
def highly_correlated_pi(df, ai):
    # correlation between Active Interests and all features
    correlation = df[features+[ai]].corr()[ai].abs()
    n = zip(correlation.nlargest(3).keys(), correlation.nlargest(3))
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    key, value = nlargest[0]
    return key

def correlated(df, ai):
    # correlation between Active Interests and all features
    topk = 10+1
    correlation = df[list(set(features)-set(zone_columns))+[ai]].corr()[ai].abs()
    correlation_real = df[list(set(features)-set(zone_columns))+[ai]].corr()[ai]
    n_real = zip(correlation.keys(), map(lambda x:round(x, 4), correlation))
    n = zip(correlation.nlargest(topk).keys(), correlation.nlargest(topk))
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    return nlargest, n_real

def sentiment_correlated(df, ai):
    # correlation between Active Interests and all features
    topk = len(zone_columns_pp)
    correlation = df[zone_columns_pp+[ai]].corr()[ai]
    n = zip(correlation.nlargest(topk).keys(), map(lambda x:round(x, 4), correlation.nlargest(topk)) )
    # avoid correlation = 1
    nlargest = [ (x,y) for x,y in n if x!=ai ]
    return nlargest

def fiveZ(all_df):
    # 3 polarity and 2 subjectivity
    all_df['zone_polarity'] = 0
    all_df['zone_subjectivity'] = 0
    all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] < 0)) ] = 'neg'
    # zone neu
    all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] == 0))  ] = 'neu'
    # zone pos
    all_df['zone_polarity'].loc[ ((all_df['PostTextPolarity'] > 0)) ] = 'pos'
    # 2 subjectivity
    # zone sub
    all_df['zone_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0)) ] = 'sub'
    # zone obj
    all_df['zone_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] == 0))  ] = 'obj'

    zone_dummies_pol = pd.get_dummies(all_df['zone_polarity'])
    zone_dummies_sub = pd.get_dummies(all_df['zone_subjectivity'])
    zone_dummies = zone_dummies_pol.join(zone_dummies_sub)
    dict_aggr = {'UserId':np.min, 'neg':np.sum, 'neu':np.sum, 'pos':np.sum, 'sub':np.sum, 'obj':np.sum}
    return zone_dummies, dict_aggr

def entropy_features(df, ent_polarity, ent_subjectivity):
    # entropy, kalo udah percentage
    total_tes = 0
    df['entropy'] = 0
    df['entropy_polarity'] = 0
    df['entropy_subjectivity'] = 0
    for x in ent_polarity:
        df[x+'_log2'] = np.log2(df[x])
        df = df.replace(-np.inf, 0)
        df['entropy_polarity'] += -(df[x]) * df[x+'_log2']
    for y in ent_subjectivity:
        df[y+'_log2'] = np.log2(df[y])
        df = df.replace(-np.inf, 0)
        df['entropy_subjectivity'] += -(df[y]) * df[y+'_log2']
    entropy_column_name = [x+'_log2' for x in ent_polarity]+[y+'_log2' for y in ent_subjectivity]+['entropy_polarity', 'entropy_subjectivity']
    return df #[entropy_column_name]

def zonings(all_df, type):
    all_df['zone'] = 0
    if type == '4p':
        # 0 as positive , best
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

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextPolarity'] <= 0.5)) ] = 'obj_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'subj_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone'])

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        dict_aggr= { 'po':np.sum, 'ps':np.sum, 'ngo':np.sum, 'ngs':np.sum, 'neg_ratio':np.mean, 'pos_ratio':np.mean, 'obj_ratio':np.mean, 'subj_ratio':np.mean }
        ent_polarity = [ 'neg_ratio', 'pos_ratio' ]
        ent_subjectivity = [ 'obj_ratio', 'subj_ratio']
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        get_ents = entropy_features(aggr_df, ent_polarity, ent_subjectivity)
        # print 'get_ents ', get_ents.dtypes
        print 'aggr ', aggr_df

        # aggr_df.drop(['entropy', 'neg_ratio_log2'], axis=1, inplace=True)
        # aggr_df = aggr_df.join(get_ents)

    elif type == '4n':
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

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] <= 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextPolarity'] <= 0.5)) ] = 'obj_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'subj_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone'])

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        dict_aggr= { 'po':np.sum, 'ps':np.sum, 'ngo':np.sum, 'ngs':np.sum, 'neg_ratio':np.mean, 'pos_ratio':np.mean, 'obj_ratio':np.mean, 'subj_ratio':np.mean }
        ent_array = [ 'neg_ratio', 'pos_ratio', 'obj_ratio', 'subj_ratio' ]
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        ents = entropy_features(aggr_df, ent_array)
        aggr_df.drop(['entropy', 'neg_ratio_log2'], axis=1, inplace=True)
        aggr_df = aggr_df.join(ents)

    elif type == '6':
        # 6 Zones
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

        all_df['ent_polarity'] = 0
        all_df['ent_subjectivity'] = 0
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] < 0) & (all_df['PostTextPolarity'] >= -1.0)) ] = 'neg_ratio'
        all_df['ent_polarity'].loc[ (all_df['PostTextPolarity'] == 0) ] = 'neu_ratio'
        all_df['ent_polarity'].loc[ ((all_df['PostTextPolarity'] > 0) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'pos_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] >= 0) & (all_df['PostTextPolarity'] <= 0.5)) ] = 'obj_ratio'
        all_df['ent_subjectivity'].loc[ ((all_df['PostTextSubjectivity'] > 0.5) & (all_df['PostTextPolarity'] <= 1.0)) ] = 'subj_ratio'

        ent_dummies = pd.get_dummies(all_df['ent_polarity']).join(pd.get_dummies(all_df['ent_subjectivity']))
        zone_dummies = pd.get_dummies(all_df['zone'])

        all_df = all_df.join(zone_dummies)
        all_df = all_df.join(ent_dummies)

        dict_aggr = { 'po':np.sum, 'ps':np.sum, 'nto':np.sum, 'nts':np.sum, 'ngo':np.sum, 'ngs':np.sum, 'neg_ratio':np.mean, 'neu_ratio':np.mean, 'pos_ratio':np.mean, 'obj_ratio':np.mean, 'subj_ratio':np.mean }
        ent_array = [ 'neg_ratio', 'neu_ratio', 'pos_ratio', 'obj_ratio', 'subj_ratio' ]
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
        # add entropy features
        ents = entropy_features(aggr_df, ent_array)
        aggr_df.drop(['entropy', 'neg_ratio_log2'], axis=1, inplace=True)
        aggr_df = aggr_df.join(ents)
    return aggr_df

# Load the text dataset, drop null non-text, assign UserId
all_df = ld.all_post()
all_df = all_df[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity']].dropna()
all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)

# Zonings and Aggregate into a user-level
aggr_df = zonings(all_df, '4p')
# aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr_sum)

# all_df['Interactions'] = all_df['LikesCount']+all_df['SharesCount']+all_df['CommentsCount']
# log_sentiment = ['LogPositivePosts','LogNeutralPosts','LogNegativePosts','LogSubjectivePosts','LogObjectivePosts']

# Load golden standard file
gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)

# Merge zone file and golden standard
aggr_df = pd.merge(aggr_df, gs_df, how='inner', left_on='UserId', right_on='UserNum')

# Separate golden standard column name
target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
aggr_df = aggr_df.join(target_dummies)
target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

# print aggr_df.head()
# zone_columns = [ x for x in dict_aggr_sum if x != 'UserId' ]

# List of Behavior Features
behavior = ['Friends','NoPhotos','NoProfilePhotos','NoCoverPhotos','NoPosts','PostTextLengthMedian','SharedNewsSum','UploadVideoSum','UploadPhotoSum','Interaction', 'GenderCode']
log_behavior = ['LogFriends','LogProfilePhotos','LogCoverPhotos','LogPosts','LogPostTextLengthMedian','LogSharedNewsSum','LogUploadVideo','LogUploadPhoto','LogInteraction', 'GenderCode']
pi = ['Arts_and_Entertainment', 'Business_and_Services', 'Community_and_Organizations', 'Education_y', 'Health_and_Beauty', 'Home_Foods_and_Drinks', 'Law_Politic_and_Government','News_and_Media', 'Pets_And_Animals', 'Recreation_and_Sports', 'Regional', 'Religion_and_Spirituality', 'Restaurants_and_Bars', 'Technology_Computers_and_Internet', 'Transportation_Travel_and_Tourism']
binary_pi = ['split_Arts_and_Entertainment','split_Business_and_Services','split_Community_and_Organizations','split_Education_y','split_Health_and_Beauty','split_Home_Foods_and_Drinks','split_Law_Politic_and_Government','split_News_and_Media','split_Pets_And_Animals','split_Recreation_and_Sports','split_Regional','split_Religion_and_Spirituality','split_Restaurants_and_Bars','split_Technology_Computers_and_Internet','split_Transportation_Travel_and_Tourism']
entropy = ['entropy']

# Preprocessing Sentiment Features
# x_df, zone_columns_pp = log_modified(aggr_df[zone_columns])
# x_df, zone_columns_pp = robust_modified(aggr_df[zone_columns])
# x_df, zone_columns_pp = percentage_modified(aggr_df[zone_columns])
# aggr_df = aggr_df.join(x_df[zone_columns_pp])

# Feature collection
# print aggr_df.dtypes
# quit()

features = zone_columns_pp + ['entropy'] # + log_behavior # log_behavior #[] #pi #binary_pi #log_behavior #behavior + pi

# Train multiple classifier
models = [LogisticRegression(), RandomForestClassifier(n_estimators=100), GaussianNB(), DecisionTreeClassifier(), SVC()]
names = ["Logistic Regression", "Random Forests Classifier", "Naive Bayes", "Decision Tree", "SVC"]
# names = ["Naive Bayes", "SVC"]
gabung = zip(models, names)

# task 2 : gimana caranya menilai feature importance ?
numCV = 3
print numCV,'-fold CV'
print names
for model, name in gabung:
    print name
    print features
    print 'MEAN ACCURACY \t MACRO F1 \t TARGET'
    for i, ai in enumerate(target):
        # to select most correlated features
        if any('Friends' in x for x in features):
            corr = correlated(aggr_df, ai)[0]
            arr_corr = [ x for (x, y) in corr ]
            correlation_value = correlated(aggr_df, ai)[1]
            if 'log_' in features:
                cleaned_features = list(set(features)-set(log_behavior)) + arr_corr
            else:
                cleaned_features = list(set(features)-set(behavior)) + arr_corr
        else:
            # tanpa hitung correlation entropy
            # untuk melihat sentiment fature yang correlated
            corr = sentiment_correlated(aggr_df, ai)
            arr_corr = [ x for (x, y) in corr ]
            correlation_value = corr
            cleaned_features = features
        """
        if 'Arts_and_Entertainment' in features:
            high_corr = highly_correlated_pi(aggr_df, ai)
        else:
            high_corr = ''
        """
        # print high_corr
        # cleaned_features = [ x for x in features if x != high_corr ]
        # print 'cleaned features: ', cleaned_features

        # correlation1 = aggr_df[features+[ai]].corr()[ai]
        # print correlation1
        print 'cleaned features : ', cleaned_features
        cv_model_accuracy = cross_val_score(model, aggr_df[cleaned_features], aggr_df[target][ai], cv=numCV, scoring='accuracy')
        cv_model_f1_macro = cross_val_score(model, aggr_df[cleaned_features], aggr_df[target][ai], cv=numCV, scoring='f1_macro')

        """
        # to find feature_importance in RF
        if name == 'Random Forests Classifier':
            rf = RandomForestClassifier()
            rf.fit(aggr_df[cleaned_features], aggr_df[target][ai])
            # rounded into 4 digits
            round_feature_important = map(lambda x:round(x, 4), rf.feature_importances_)
            importance_features = sorted(zip(round_feature_important, cleaned_features), reverse=True)
        else:
            # kalo metode lain gimana ya?
            pass
        """
        # print 'PREDICTED GROUP: ', v
        # print 'MEAN ACCURACY \t MACRO F1: ',
        round_accuracy = round(cv_model_accuracy.mean(), 3)
        round_f1macro = round(cv_model_f1_macro.mean(), 3)
        # print cv_model_accuracy.mean(), '\t', cv_model_f1_macro.mean(), '\t', ai
        print round_accuracy, '\t', round_f1macro, '\t', ai, '\t', cleaned_features, '\t', correlation_value
    print '\n'