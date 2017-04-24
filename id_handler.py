from Utils import list2dict


# 构造id类特征
def build_id_features(stat):
    adIDs = list2dict(stat[3].unique().tolist())
    aderIDs = list2dict(stat[4].unique().tolist())
    keywordIDs = list2dict(stat[8].unique().tolist())
    userIDs = list2dict(stat[11].unique().tolist())

    querydf = stat[7].value_counts()
    querylist = []
    for i, row in querydf.iteritems():
        if (int(row) > 20):                 # 只选择频率超过20的, 剩下的记为unknown
            querylist.append(i)
    queryIDs = list2dict(querylist)

    titledf = stat[9].value_counts()
    titlelist = []
    for i, row in titledf.iteritems():
        if (int(row) > 20):                 # 只选择频率超过20的, 剩下的记为unknown
            titlelist.append(i)
    titleIDs = list2dict(titlelist)
    print "Building id finished."
    return adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs
