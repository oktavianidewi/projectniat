import random
import numpy as np
import pandas as pd
import load_all_dataset as ld
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from operator import add, truediv
import json

"""
----------------------------------------------------
                pseudo code of GA
----------------------------------------------------
Initialize Population
While (not met Termination criteria):
    Selection (Roulette Wheel)
    Reproduction (Cross-over, one portion crossover)
    Replacement
End-While
"""

def polarity_zones():
    polarity_div = 2 #4 #2 #4 # selain 0
    sentiment_div = 2
    # asumsi sentiment div 0 - 0.5 dan 0.5 - 1.0

    # generate number of division which the summing equal to 1
    pos = np.random.dirichlet(np.ones(polarity_div),size=1)
    # neg = -(pos)
    neg = -(np.random.dirichlet(np.ones(polarity_div),size=1))

    all = np.append(neg, pos)
    all = sorted( np.insert(all, polarity_div, 0) )
    zone = []
    for i in all: zone.append(round(i, 3))
    # print zone

    # zone = [-0.453, -0.416, -0.114, -0.017, 0.0, 0.017, 0.114, 0.416, 0.453]

    polarity_zone = []
    polarity_threshold_next = 0
    for i, x in enumerate(zone):
        polarity_threshold = x
        if polarity_threshold >= 0:
            if i < len(zone)-1:
                polarity_threshold_current = polarity_threshold_next
                polarity_threshold_next += zone[i+1]
                if i == len(zone)-1:
                    if round(polarity_threshold_next, 3) < 1.0:
                        polarity_threshold_next = 1.0
                    else:
                        polarity_threshold_next = round(polarity_threshold_next, 3)
                else:
                    polarity_threshold_next = round(polarity_threshold_next, 3)

                polarity_zone.append([round(polarity_threshold_current, 3), polarity_threshold_next])

    polarity_zone = [ [-x, -y] for x,y in polarity_zone ] + [[0.0, 0.0]] + polarity_zone

    # print 'subjectivity zones ', subjectivity_zone
    # print 'polarity zones ', polarity_zone


    return polarity_zone

import time
def writeToFile(result):
    date = time.strftime('%Y%m%d%H',time.localtime(time.time()))
    filename = 'hasil_'+date+'.json'
    print filename
    # harus ada pengecekan fileexist atau ga
    file = open(filename, "w+")
    file.write(json.dumps(result))
    file.close()
    return True

def individual():
    return polarity_zones()

def population(noOfIndividual):
    return [individual() for x in range(noOfIndividual)]

def threshold_zones(all_df, polarity_zone, subjectivity_zone):
    for index_sub, subjectivity in enumerate(subjectivity_zone):
        # print subjectivity
        if index_sub < len(subjectivity_zone) - 1:
            for index_pol, polarity in enumerate(polarity_zone):
                # print index_pol, polarity
                nama_zone = str(index_sub) + '_' + str(index_pol)
                if polarity[1] < 0.0:
                    # jika polarity < 0
                    all_df['zone'].loc[
                        ((all_df['PostTextPolarity'] < polarity[0]) & (all_df['PostTextPolarity'] >= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                        all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone
                elif polarity[1] == 0.0:
                    # jika polarity = 0
                    all_df['zone'].loc[(all_df['PostTextPolarity'] == 0.0) &
                                       ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                                       all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone
                else:
                    all_df['zone'].loc[
                        ((all_df['PostTextPolarity'] > polarity[0]) & (all_df['PostTextPolarity'] <= polarity[1])) &
                        ((all_df['PostTextSubjectivity'] >= subjectivity_zone[index_sub]) & (
                        all_df['PostTextSubjectivity'] <= subjectivity_zone[index_sub + 1]))] = nama_zone
    return all_df

def dataset(type, polarity_zone, subjectivity_zone):
    if type == 'old':
        # Load the text dataset, drop null non-text, assign UserId
        all_df = ld.all_post()
        all_df = all_df[['postId', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity', 'PostTextSubjectivity']].dropna()
        all_df['UserId'] = all_df['postId'].str.split('_').str.get(0).astype(int)
        all_df['zone'] = 'a'

        all_df = threshold_zones(all_df, polarity_zone, subjectivity_zone)

        zone_dummies = pd.get_dummies(all_df['zone'])
        column_zone_dummies = zone_dummies.columns.tolist()
        dict_aggr = {x:np.sum for x in column_zone_dummies}
        all_df = all_df.join(zone_dummies)
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()

        # Load golden standard file
        gs_df = pd.read_csv('data/userlevel_all_features_1007.csv', header=0)

        # Merge zone file and golden standard
        aggr_df = pd.merge(aggr_df, gs_df, how='inner', left_on='UserId', right_on='UserNum')
        aggr_df = aggr_df[['UserId', 'ActiveInterests']+column_zone_dummies]

        # harusnya digabung dulu, baru activeinterests dibikin dummies
    elif type == 'new':
        # Load the new dataset, drop null non-text, assign UserId
        ds_file_array = ['data/english_foodgroup_new.json', 'data/english_TEDtranslate_new.json',
                         'data/english_traveladdiction_new.json']
        all_df = ld.new_dataset(ds_file_array)
        all_df = all_df[['UserID', 'LikesCount', 'SharesCount', 'CommentsCount', 'PostTextLength', 'PostTextPolarity',
                         'PostTextSubjectivity', 'ActiveInterests']].dropna()
        all_df['UserId'] = all_df['UserID']

        all_df['zone'] = 'a'
        all_df = threshold_zones(all_df, polarity_zone, subjectivity_zone)

        zone_dummies = pd.get_dummies(all_df['zone'])
        column_zone_dummies = zone_dummies.columns.tolist()
        dict_aggr = {x: np.sum for x in column_zone_dummies}
        dict_aggr.update({'ActiveInterests': np.min})
        all_df = all_df.join(zone_dummies)
        aggr_df = all_df.groupby(['UserId'], sort=True).agg(dict_aggr).reset_index()
    """
    target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
    aggr_df = aggr_df.join(target_dummies)
    target = sorted(list(set(target_dummies.columns.tolist()) - set(['Random'])))
    """

    return aggr_df, column_zone_dummies #, target


# belum di edit
def fitness(individu):
    # rumus F1
    # print individu
    polarity_zone = individu
    subjectivity_zone = [0.0, 0.5, 1.0]

    aggr_df_old, column_zone_dummies = dataset('old', polarity_zone, subjectivity_zone)
    aggr_df_new, column_zone_dummies_new = dataset('new', polarity_zone, subjectivity_zone)
    cols = list(aggr_df_old.columns.values)
    aggr_df = aggr_df_old.append(aggr_df_new[cols], ignore_index=True)

    # Separate golden standard column name
    target_dummies = pd.get_dummies(aggr_df['ActiveInterests'])
    # print 'target dummies ', target_dummies
    aggr_df = aggr_df.join(target_dummies)
    target = sorted( list( set( target_dummies.columns.tolist() ) - set( ['Random'] ) ) )

    print 'total data: ', len(aggr_df)

    models = [GaussianNB()]
    names = ["Gaussian Naive Bayes"]
    # models = [LogisticRegression()]
    # names = ["Logistic Regression"]
    gabung = zip(models, names)
    numCV = 5
    fscore_arr = []
    for model, name in gabung:
        # print name
        for i, ai in enumerate(target):
            # cari F1 score
            # print ai
            fscore_arr.append( cross_val_score(model, aggr_df[column_zone_dummies], aggr_df[target][ai], cv=numCV, scoring='f1_macro') )
            # round_feature_important = map(lambda x:round(x, 4), rf.feature_importances_)
            # map(lambda x:round(x, 3), correlation)
        # print fscore_arr
    rerataf1 = round(np.mean(fscore_arr), 3)
    return rerataf1

def selection(populasi):
    # call fitness
    fitness_populasi = [ fitness(i) for i in populasi ]
    total_probability = reduce(add, fitness_populasi, 0)
    probabilitas_populasi = [ round(truediv(fitness_individu, total_probability), 3) for fitness_individu in fitness_populasi ]
    """
    print 'populasi ', populasi
    print 'fitness ', fitness_populasi
    print 'probabilitas populasi', probabilitas_populasi
    """
    # ngga pake di-inverse
    return probabilitas_populasi

def roulettewheel(populasi):
    # call selection
    probabilitas_populasi = selection(populasi)
    kumulatif_probabilitas = [ reduce(add, probabilitas_populasi[:i], 0) for i in range(len(probabilitas_populasi)+1) ]
    randomnum = [random.uniform(0.0, 1.0) for p in range(0, len(populasi))]

    """
    print probabilitas_populasi
    print kumulatif_probabilitas
    print randomnum
    """

    populasi_fittest = []
    probabilitas_populasi_fittest = []

    # roulette wheel
    for j in range(len(randomnum)):
        # print j
        # print randomnum[j]
        for k in range(len(kumulatif_probabilitas)-1):
            if (randomnum[j] > kumulatif_probabilitas[k]) and (randomnum[j] < kumulatif_probabilitas[k+1]):
                populasi_fittest.append( (populasi[k], probabilitas_populasi[k]) )
    sorted_populasi_fittest = sorted(populasi_fittest, key=lambda x:x[1], reverse=True)
    populasi_fittest = [ h[0] for h in sorted_populasi_fittest ]
    # print 'gabung ', sorted_populasi_fittest
    return populasi_fittest

def crossover(populasi_fittest, co_rate):
    crossover_portion = int(co_rate * len(populasi_fittest))
    parents_portion = len(populasi_fittest) - crossover_portion
    # single crossover
    parents = []
    new_individu = []
    for i in range(0, crossover_portion, 2):
        individu = populasi_fittest[i]
        position_to_crossover = individu.index([0.0, 0.0])
        father = populasi_fittest[i]
        mother = populasi_fittest[i+1]
        anak1 = father[:position_to_crossover]+mother[position_to_crossover:]
        anak2 = mother[:position_to_crossover]+father[position_to_crossover:]
        new_individu.append(anak1)
        new_individu.append(anak2)
        parents.append(father)
        parents.append(mother)

    # replace the father n mother in
    # new_individu =  populasi_fittest + new_individu
    # print 'new ind arr ', new_individu
    return new_individu, parents[:parents_portion]


# percobaan pada single generation
# duluan roulettewheel, baru crossover

"""
population = population(4)
print population
# roulette wheel selection
fittest_population = roulettewheel(population)
print fittest_population
# crossover
parents, new_individu = crossover(fittest_population, 0.5)
new_individu = parents + new_individu
print new_individu
quit()
"""

import datetime
population_history = []
generasi = 20
for i in range(generasi):
    print 'generasi-', i, datetime.datetime.now()
    if i == 0:
        # pake sample populasi awal
        population = population(100)
    else:
        # pake hasil iterasi
        population = parents + new_individu

    population_history.append(population)
    fittest_population = roulettewheel(population)
    new_individu, parents = crossover(fittest_population, 0.8)
    print population

# simpan population history in a json file

writeToFile(population_history)

quit()
for i in range(len(population_history)):
    print 'generasi ke-', i
    hitung_fitness = [ fitness(item_history_populasi) for item_history_populasi in population_history[i] ]
    index_fittest = hitung_fitness.index(max(hitung_fitness))
    print hitung_fitness
    print population_history[i]
    print population_history[i][index_fittest]
