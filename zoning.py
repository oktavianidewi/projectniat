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
    normal_df = (df[columns] - df[columns].min())/(df[columns].max() - df[columns].min())
    normal_df = normal_df.join(df[["UserId"]+target])
    return normal_df

def scaler_modified(df):
    # columns = ['po', 'ps', 'nto', 'nts', 'ngo', 'ngs']
    standard_scaler = StandardScaler()
    scaled_df = pd.DataFrame(standard_scaler.fit_transform(df[columns]), columns=[columns])
    scaled_df = scaled_df.join(df[["UserId"]+target])
    return scaled_df

def robust_modified(df):
    robust_scaler = RobustScaler()
    robust_df = pd.DataFrame(robust_scaler.fit_transform(df[columns]), columns=[columns])
    robust_df = robust_df.join(df[["UserId"]+target])
    return robust_df

def percentage(df):
    return (df - df.min())/(df.max() - df.min())

def smoothing():
    # piye carane
    pass

# LOAD THE DATASET
all_df = ld.all_post()
# merge with post clustered result
all_df = all_df[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity']].dropna()

all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)

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

# load golden standard
gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)
gs_df = gs_df[['UserNum', 'DataCollectionCat', 'ActiveInterests']]
all_df = all_df.join(zone_dummies)

# aggregation type
# dict_aggr_mean = {'UserId':np.min, 'po':np.mean, 'ps':np.mean, 'nto':np.mean, 'nts':np.mean, 'ngo':np.mean, 'ngs':np.mean}
# dict_aggr_sum = {'UserId':np.min, 'po':np.sum, 'ps':np.sum, 'nto':np.sum, 'nts':np.sum, 'ngo':np.sum, 'ngs':np.sum}
dict_aggr_mean = {'UserId':np.min, 'po':np.mean, 'ps':np.mean, 'ngo':np.mean, 'ngs':np.mean}
dict_aggr_sum = {'UserId':np.min, 'po':np.sum, 'ps':np.sum, 'ngo':np.sum, 'ngs':np.sum}
aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr_sum)

print aggr_df
quit()

# merge all df and golden standard
aggr_df = pd.merge(aggr_df, gs_df, how='inner', left_on='UserId', right_on='UserNum')
aggr_df = aggr_df.drop('UserNum', axis=1)

# separate golden standard column name
target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
aggr_df = aggr_df.join(target_dummies)
target = target_dummies.columns.tolist()

columns = aggr_df.columns.tolist()
columns = [c for c in columns if c not in ["ActiveInterests", "DataCollectionCat", "UserId"] + target]

aggr_df = robust_modified(aggr_df)
"""
# normalize features
aggr_df = normalize(aggr_df)
aggr_df = scaler_modified(aggr_df)

"""

# aggr_df = robust_modified(aggr_df)

"""
valid option untuk cv model score
['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
"""

# multiple classifier
models = [LogisticRegression(), RandomForestClassifier(n_estimators=100), GaussianNB(), DecisionTreeClassifier(), SVC()]
names = ["Logistic Regression", "Random Forests Classifier", "Naive Bayes", "Decision Tree", "SVC"]
gabung = zip(models, names)

for model, name in gabung:
    print name
    for i, v in enumerate(target):
        cv_model_accuracy = cross_val_score(model, aggr_df[columns], aggr_df[target][v], cv=5, scoring='accuracy')
        cv_model_f1_macro = cross_val_score(model, aggr_df[columns], aggr_df[target][v], cv=5, scoring='f1_macro')

        print 'PREDICTED GROUP: ', v
        print 'MEAN ACCURACY \t MACRO F1: ', cv_model_accuracy.mean(), '\t', cv_model_f1_macro.mean()
    print '\n'

quit()

train = aggr_df.sample(frac=0.7, random_state=1)
test = aggr_df.loc[~aggr_df.UserId.isin(train.UserId)]

# single classifier
for i, v in enumerate(target):
    logReg.fit(train[columns], train[target][v])
    logReg_prediction_result = logReg.predict(test[columns])

    """
    # single testing
    logReg_error_rate = mean_squared_error(logReg_prediction_result, test[target][v])
    logReg_accuracy = accuracy_score(test[target][v], logReg_prediction_result)
    print 'error logReg : ', logReg_error_rate
    print 'accuracy logReg : ', logReg_accuracy
    """

    cv_model_accuracy = cross_val_score(logReg, aggr_df[columns], aggr_df[target][v], cv=5, scoring='accuracy')
    cv_model_f1_macro = cross_val_score(logReg, aggr_df[columns], aggr_df[target][v], cv=5, scoring='f1_macro')

    print v
    print 'mean accuracy: ', cv_model_accuracy.mean()
    print 'mean f1 macro: ', cv_model_f1_macro.mean()