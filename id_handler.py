# -*- coding:utf-8 -*-
from Utils import list2dict
import pandas as pd
import constants
import numpy as np

# 构造id类特征
def build_id_features(stat):
    # 只选择频率较高的, 剩下的记为unknown, 过滤掉90%的id

    # ad
    addf = stat[3].value_counts()
    adlist = []
    for i, row in addf.iteritems():
        if (int(row) > 5):
            adlist.append(i)
    adIDs = list2dict(adlist)

    # ader
    aderdf = stat[4].value_counts()
    aderlist = []
    for i, row in aderdf.iteritems():
        if (int(row) > 5):
            aderlist.append(i)
    aderIDs = list2dict(aderlist)

    # keyword
    keyworddf = stat[8].value_counts()
    keywordlist = []
    for i, row in keyworddf.iteritems():
        if (int(row) > 5):
            keywordlist.append(i)
    keywordIDs = list2dict(keywordlist)

    # user
    userdf = stat[11].value_counts()
    userlist = []
    for i, row in userdf.iteritems():
        if (int(row) > 5):
            userlist.append(i)
    userIDs = list2dict(userlist)


    # query
    querydf = stat[7].value_counts()
    querylist = []
    for i, row in querydf.iteritems():
        if (int(row) > 10):
            querylist.append(i)
    queryIDs = list2dict(querylist)

    # title
    titledf = stat[9].value_counts()
    titlelist = []
    for i, row in titledf.iteritems():
        if (int(row) > 10):
            titlelist.append(i)
    titleIDs = list2dict(titlelist)
    print "Building id finished."
    return adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs


# 统计id类特征
def statistic():
    stat = pd.read_csv(constants.dir_path + "sample\\training.part", header=None, delimiter='\t', dtype=str)
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
        print "ad 20%: " + str(np.percentile(a, 85))  # 10%分位数

        querylist = []
        querydf = stat[4].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "ader 20%: " + str(np.percentile(a, 85))  # 10%分位数

        querylist = []
        querydf = stat[8].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "keyword 20%: " + str(np.percentile(a, 85))  # 10%分位数

        querylist = []
        querydf = stat[11].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "user 20%: " + str(np.percentile(a, 85))  # 10%分位数

        querylist = []
        querydf = stat[7].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "query 20%: " + str(np.percentile(a, 85))  # 10%分位数

        querylist = []
        querydf = stat[9].value_counts()
        for i, row in querydf.iteritems():
                querylist.append(int(row))
        a = np.array(querylist)
        print "title 20%: " + str(np.percentile(a, 85))  # 10%分位数

if __name__ == '__main__':
    statistic()