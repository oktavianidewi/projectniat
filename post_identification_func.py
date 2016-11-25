import json
import collections
import numpy as np
import scipy.stats as stats
import pylab as pl
from datetime import datetime

def normalDist(arraydata):
    h = sorted(arraydata)  #sorted
    print h
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    pl.plot(h, fit,'-o')
    pl.hist(h, facecolor="Yellow", alpha = 0.75, normed=True) #use this to draw histogram of your data
    pl.title(r'$\mathrm{Histogram\ of\ Average Posts/Month:}\ \mu=%s,\ \sigma=np.std(h)$')
    # pl.title(r'$\mathrm{Histogram\ of\ Average Posts/Month:}\ \mu='np.mean(h)',\ \sigma='+np.std(h)+'$')
    pl.xlabel("Average Posts/Month")
    pl.ylabel("Frequency")
    pl.grid(True)
    pl.show()


def dailyActivitiesPost(postTitle):
    countDailyAct = 0
    keywords = [
        'traveling' ,
        'reading' , 'watching', 'walking', 'looking for', 'ran', 'completed', 'walked', 'biked',
        'listening', 'eating', 'getting', 'checked in', 'making', 'celebrating', 'playing',
        'reviewed', 'thinking', 'recommends',
        'added ', # without photo dan video keyword,
        'uploaded a new', # without photo dan video keyword,
        'at ', 'with ', 'in '
    ]

    for i in keywords:
        if ( i in postTitle and 'photo' not in postTitle ) and ( i in postTitle and 'video' not in postTitle ) and \
                ( i in postTitle and 'feeling' not in postTitle ) and ( i in postTitle and 'shared \'s' not in postTitle ) :
            countDailyAct = 1
    # print countDailyAct, postTitle
    return countDailyAct

def sharedNews(postTitle):
    countSharedNews = 0
    keywords = ['via', 'shared \'s', 'shared a', 'commented on', 'likes', 'published']
    for i in keywords:
        if ( i in postTitle and 'photo' not in postTitle) and ( i in postTitle and 'video' not in postTitle):
            countSharedNews = 1
    return countSharedNews

def other(postTitle, sharedLink):
    resultOther = []
    countUploadedVideo = 0
    countSharedNews = 0

    keywords = ['???to?', '~']

    # print 'shared link : ', sharedLink
    for i in keywords:
        if ( i in postTitle and 'photo' not in postTitle) and ( i in postTitle and 'video' not in postTitle):
            # www.reverbnation.com u/ lagu
            if 'youtube' in sharedLink:
                resultOther = ['countUploadedVideo', 1]
                # '~' punya ShreLinkSource dan dari youtube.com -> tambah ke countUploadedVideo
            elif 'youtube' not in sharedLink:
                resultOther = ['countSharedNews', 1]
                # '~' punya ShreLinkSource dan ShreLinkSource -> tambah ke countSharedNews
            else:
                resultOther = ['countSharedNews', 0]
        else:
            resultOther = ['no', 0]
    return resultOther

def sharedFeeling(postTitle):
    countSharedFeeling = 0
    keywords = ['feeling', 'shared a memory']
    for i in keywords:
        if ( i in postTitle and 'photo' not in postTitle) and ( i in postTitle and 'video' not in postTitle) and \
                ( i in postTitle and 'added ' not in postTitle) and ( i in postTitle  and 'shared ' not in postTitle) :
            countSharedFeeling = 1
    return countSharedFeeling

def uploadedPhoto(postTitle):
    countUploadedPhoto = 0
    keywords = ['new photos to the album', 'profile picture', 'cover photo', 'a new photo']
    for i in keywords:
        if ( i in postTitle ):
            countUploadedPhoto = 1
    # print countUploadedPhoto, postTitle
    return countUploadedPhoto

# ~ bisa jadi shared link, definisian fungsi baru
# like an articles -> sharedNews
# comment -> commentPost
"""
def sharedNews(postTitle):
    countSharedNews = 0
    keywords = ['']

    return countSharedNews
"""
def uploadedVideo(postTitle):
    countUploadedVideo = 0
    keywords = ['video', 'added a new video', 'shared a video', 'made a video']
    for i in keywords:
        if ( i in postTitle ):
            countUploadedVideo = 1
    return countUploadedVideo

