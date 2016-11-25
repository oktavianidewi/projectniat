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

# LOAD THE DATASET
# load golden standard
# pd.set_option('display.max_rows', 120)
df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)
# LYH
lyh_ptg = ['SentimentPer33_pgt', 'SentimentPer22_pgt', 'SentimentPer11_pgt', 'SentimentPer00_pgt', 'SentimentPer_11_pgt']
lyh_ptg_log = ['LogSentimentPer33_pgt', 'LogSentimentPer22_pgt', 'LogSentimentPer11_pgt', 'LogSentimentPer00_pgt', 'LogSentimentPer_11_pgt']

pol3sub2_log = ['LogPositivePosts', 'LogNeutralPosts', 'LogNegativePosts', 'LogSubjectivePosts', 'LogObjectivePosts']

# Cluster 5
cluster5_sum_pgt = ['resultCluster5_0_pgt', 'resultCluster5_1_pgt', 'resultCluster5_2_pgt', 'resultCluster5_3_pgt', 'resultCluster5_4_pgt']
cluster5_sum = ['resultCluster5_0_sum', 'resultCluster5_1_sum', 'resultCluster5_2_sum', 'resultCluster5_3_sum', 'resultCluster5_4_sum']
cluster5_sum_log = ['LogResultCluster5_0_sum', 'LogResultCluster5_1_sum', 'LogResultCluster5_2_sum', 'LogResultCluster5_3_sum', 'LogResultCluster5_4_sum']

# Cluster 4
cluster4_sum = ['Cluster_2', 'Cluster_3', 'Cluster_0', 'Cluster_1']
cluster4_sum_ptg = ['Cluster_2_percent', 'Cluster_3_percent', 'Cluster_0_percent', 'Cluster_1_percent']
cluster4_sum_log = ['LogCluster_2', 'LogCluster_1', 'LogCluster_3', 'LogCluster_0']

columns = lyh_ptg_log
df = df[['UserNum', 'DataCollectionCat', 'ActiveInterests'] + columns]
# separate golden standard column name
target_dummies = pd.get_dummies(df['ActiveInterests'])
target = target_dummies.columns.tolist()

df[columns] = df[columns].convert_objects(convert_numeric=True)
df[columns] = df[columns].fillna(0)
# print df[columns].isnull().any()
# print df[columns].dtypes
# print df[columns].describe()
# df[columns] = df[columns].astype(float)

df = df.join(target_dummies)

"""
valid option untuk cv model score
['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']
"""

# multiple classifier
models = [LogisticRegression(), RandomForestClassifier(n_estimators=100), GaussianNB(), DecisionTreeClassifier(), SVC()]
names = ["Logistic Regression", "Random Forests Classifier", "Naive Bayes", "Decision Tree", "SVC"]
gabung = zip(models, names)

numCV = 3
print numCV, 'cv'
print names
print columns
for model, name in gabung:
    print name
    print 'MEAN ACCURACY \t MACRO F1 \t TARGET'
    for i, v in enumerate(target):
        cv_model_accuracy = cross_val_score(model, df[columns], df[target][v], cv=numCV, scoring='accuracy')
        cv_model_f1_macro = cross_val_score(model, df[columns], df[target][v], cv=numCV, scoring='f1_macro')
        print round(cv_model_accuracy.mean(), 3), '\t', round(cv_model_f1_macro.mean(), 3), '\t', v
    print '\n'