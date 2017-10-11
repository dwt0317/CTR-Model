# -*- coding:utf-8 -*-
from Utils import list2dict
import pandas as pd
import constants
import numpy as np


# 构造id类特征
def build_id_features(stat):
    # 只选择频率较高的, 剩下的记为unknown, 过滤掉90%的id
    # default setting: 30

    threshold = 3
    # ad
    addf = stat[3].value_counts()
    adlist = []
    for i, row in addf.iteritems():
        if (int(row) > threshold):
            adlist.append(i)
    adIDs = list2dict(adlist)

    # ader
    aderdf = stat[4].value_counts()
    aderlist = []
    for i, row in aderdf.iteritems():
        if (int(row) > threshold):
            aderlist.append(i)
    aderIDs = list2dict(aderlist)

    # depth
    depthdf = stat[5].value_counts()
    depthlist = []
    for i, row in depthdf.iteritems():
        if (int(row) > threshold):
            depthlist.append(i)
    depthIDs = list2dict(depthlist)

    # position
    posdf = stat[6].value_counts()
    poslist = []
    for i, row in posdf.iteritems():
        if (int(row) > threshold):
            poslist.append(i)
    posIDs = list2dict(poslist)

    # query
    querydf = stat[7].value_counts()
    querylist = []
    for i, row in querydf.iteritems():
        if (int(row) > threshold):
            querylist.append(i)
    queryIDs = list2dict(querylist)

    # keyword
    keyworddf = stat[8].value_counts()
    keywordlist = []
    for i, row in keyworddf.iteritems():
        if (int(row) > threshold):
            keywordlist.append(i)
    keywordIDs = list2dict(keywordlist)

    # title
    titledf = stat[9].value_counts()
    titlelist = []
    for i, row in titledf.iteritems():
        if (int(row) > threshold):
            titlelist.append(i)
    titleIDs = list2dict(titlelist)

    # description
    desdf = stat[10].value_counts()
    deslist = []
    for i, row in desdf.iteritems():
        if (int(row) > threshold):
            deslist.append(i)
    desIDs = list2dict(deslist)


    # user
    userdf = stat[11].value_counts()
    userlist = []
    for i, row in userdf.iteritems():
        if (int(row) > 5):
            userlist.append(i)
    userIDs = list2dict(userlist)

    print "Building id finished."
    return adIDs, aderIDs, depthIDs, posIDs, queryIDs, keywordIDs, titleIDs, desIDs, userIDs


'''
position:1, 2, 3
depth:1, 2, 3
'''


# 统计id类特征
def statistic():
    stat = pd.read_csv(constants.dir_path + "sample\\training.part", header=None, delimiter='\t', dtype=str)
    print stat[5].value_counts()
    if False:
        print "Reading file finished."
        adIDs = list2dict(stat[3].unique().tolist())
        print "ad size: " + str(len(adIDs))
        aderIDs = list2dict(stat[4].unique().tolist())
        print "ader size: " + str(len(aderIDs))
        keywordIDs = list2dict(stat[8].unique().tolist())
        print "keyword size: " + str(len(keywordIDs))
        userIDs = list2dict(stat[11].unique().tolist())
        print "user size: " + str(len(userIDs))
        print "query size: " + str(len(stat[7].unique().tolist()))
        print "title size: " + str(len(stat[9].unique().tolist()))

    if True:
        adlist = []
        addf = stat[3].value_counts()
        for i, row in addf.iteritems():
            adlist.append(int(row))
        a = np.array(adlist)
        print a
        print "ad 20%: " + str(np.percentile(a, 10))  # 10%分位数

        querylist = []
        querydf = stat[4].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "ader 20%: " + str(np.percentile(a, 10))  # 10%分位数

        querylist = []
        querydf = stat[8].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "keyword 20%: " + str(np.percentile(a, 10))  # 10%分位数

        querylist = []
        querydf = stat[11].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "user 20%: " + str(np.percentile(a, 10))  # 10%分位数

        querylist = []
        querydf = stat[7].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "query 20%: " + str(np.percentile(a, 10))  # 10%分位数

        querylist = []
        querydf = stat[9].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "title 20%: " + str(np.percentile(a, 85))  # 10%分位数

if __name__ == '__main__':
    statistic()