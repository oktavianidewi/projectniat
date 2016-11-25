import pandas as pd
import numpy as np

filename = 'data/user_classification_fitur.csv'
df = pd.read_csv(filename, header=0)


df['coded_HJ_Dewi'] = df['HJ_Dewi'].str.strip()
df['coded_HJ_LYH'] = df['HJ_LYH'].str.strip()
dict_usertype = {'Average':'0', 'Critics':'1', 'Monologue':'2', 'Sharer':'3', 'Sociable':'4'}
df['coded_HJ_Dewi'] = df['coded_HJ_Dewi'].map(dict_usertype)
df['coded_HJ_LYH'] = df['coded_HJ_LYH'].map(dict_usertype)

df['sex'] = df['Gender']
df['sex'] = df['sex'].map({'Female':0, 'Male':1})
df['sex'].fillna('2', inplace=True)

df_dummies = pd.get_dummies(df['coded_HJ_Dewi']).astype(int)
df = df.join(df_dummies)

# fitur
arr_sentiment = ['SentimentPer33_pgt', 'SentimentPer22_pgt', 'SentimentPer11_pgt', 'SentimentPer00_pgt', 'SentimentPer_11_pgt']
df['interaction'] = df['SharesCountSum']+df['LikesCountSum']+df['CommentsCountSum']
# print df[['HJ_Dewi', 'HJ_LYH', 'coded_HJ_Dewi', 'coded_HJ_LYH']]

train = df.sample(frac=0.7, random_state=1)
test = df.loc[~df.UserNum.isin(train.UserNum)]

columns1 = arr_sentiment
columns2 = arr_sentiment + ['interaction', 'SharedNewsSum']
target1 = ['0', '1', '2', '3', '4']
# target1 = [0, 1, 2, 3, 4]
# target2 = ['coded_HJ_LYH']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, recall_score, precision_score, f1_score

quit()

lr = LogisticRegression()
# buat looping dari sini
lr.fit(train[columns1], train[target1[0]])
lr_predictions = lr.predict(test[columns1])
print lr_predictions

quit()
error_rate = mean_squared_error(lr_predictions, test[target1])

lr_accuracy_score = lr.score(train[columns1], train[target1])
lr_recall_score = recall_score(test[target1], lr_predictions)
lr_pred_score = precision_score(test[target1], lr_predictions)
lr_f1_score = f1_score(test[target1], lr_predictions)

print 'error logistic : ', error_rate
print 'accuracy : ', lr_accuracy_score
print 'prf value : ', lr_pred_score, lr_recall_score, lr_f1_score