# -*- coding:utf-8 -*-

import pandas as pd
from Utils import file_len, list2dict
import Constants

# 采样
def sample():
    file_read = open(Constants.dir_path + "training.txt")
    file_train = open(Constants.dir_path + "sample\\training.part",'w')
    file_valid = open(Constants.dir_path + "sample\\validation.part",'w')
    file_test = open(Constants.dir_path + "sample\\test.part", 'w')
    i = 0
    for line in file_read:
        # file_train.write(line)
        if i % 100000 == 0 :
            print i
        if i < 1800000:
            file_train.write(line)
        elif i >= 1800000 and i < 2000000:
            file_valid.write(line)
        else:
            file_test.write(line)
        if i > 2200000 :    # 多了一行数据
            file_read.close()
            file_train.close()
            file_valid.close()
            file_test.close()
            break
        i = i + 1


def build_Y():
    file_read = open(Constants.dir_path + "sample\\test.part")
    file_Y = open(Constants.dir_path + "sample\\test.Y", 'w')
    for line in file_read:
        y = line[:1]
        if int(y) > 0 : y = 1
        file_Y.write(str(y) + '\n')
    file_read.close()
    file_Y.close()


def ctr_helper(idset, impre_set, click_set):
    alpha = 0.05  # for smoothing
    beta = 75
    ctr_set = []
    for i in range(6):
        ids = idset[i]
        ctrs = {}
        for id in ids:
            impression = float(impre_set[i].setdefault(id, 0))
            click = float(click_set[i].setdefault(id, 0))
            ctr = (click + alpha * beta) / (impression + beta)
            ctrs[id] = round(ctr, 5)
        ctr_set.append(ctrs)
    print "Computing ctr finished."
    return ctr_set[0], ctr_set[1], ctr_set[2], ctr_set[3], ctr_set[4], ctr_set[5]


# CTR features [ad, advertiser, query, keyword, title, user]
def build_ctr(idset):
    impre_ad, impre_ader, impre_keyword, impre_user, impre_query, impre_title = {}, {}, {}, {}, {}, {}
    click_ad, click_ader, click_keyword, click_user, click_query, click_title = {}, {}, {}, {}, {}, {}

    stat_file = open(Constants.dir_path + "sample\\total.part",'r')
    for line in stat_file:            # 迭代pandas太慢了，不要用
        row = line.strip('\n').split('\t')
        impre_ad[row[3]] = 1 + impre_ad.setdefault(row[3], 0)
        impre_ader[row[4]] = 1 + impre_ader.setdefault(row[4], 0)
        impre_query[row[7]] = 1 + impre_query.setdefault(row[7], 0)
        impre_keyword[row[8]] = 1 + impre_keyword.setdefault(row[8], 0)
        impre_title[row[9]] = 1 + impre_title.setdefault(row[9], 0)
        impre_user[row[11]] = 1 + impre_user.setdefault(row[11], 0)

        if int(row[0]) == 1:
            click_ad[row[3]] = 1 + click_ad.setdefault(row[3], 0)
            click_ader[row[4]] = 1 + click_ader.setdefault(row[4], 0)
            click_query[row[7]] = 1 + click_query.setdefault(row[7], 0)
            click_keyword[row[8]] = 1 + click_keyword.setdefault(row[8], 0)
            click_title[row[9]] = 1 + click_title.setdefault(row[9], 0)
            click_user[row[11]] = 1 + click_user.setdefault(row[11], 0)
    print "Counting impression and click finished."
    impre_set = [impre_ad, impre_ader, impre_keyword, impre_user, impre_query, impre_title]
    click_set = [click_ad, click_ader, click_keyword, click_user, click_query, click_title]
    return ctr_helper(idset, impre_set, click_set)

# query-title, query-description, use start end to locate row of record
def build_similarity_features(start):
    simi_feature_file = open(Constants.dir_path + "sample\\mapping\\txtNormDistance_2.feature")
    query_title_simi = []
    query_desc_simi = []
    for line in simi_feature_file:
        tuple2 = line.strip('\n').split('\t')
        query_title_simi.append(tuple2[0])
        query_desc_simi.append(tuple2[1])
    print "similarity:" + query_title_simi[start], query_desc_simi[start]
    return query_title_simi[start:], query_desc_simi[start:]

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


# 以scipy稀疏矩阵形式存储
def build_x_helper(idset, ctr_set, user_profile, file_read, file_write, similarity_start):
    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = idset[0], idset[1], idset[2], idset[3], idset[4] \
        , idset[5]
    ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user = ctr_set[0], ctr_set[1], ctr_set[2], ctr_set[3] \
        , ctr_set[4], ctr_set[5]

    query_title_similarity, query_desc_similarity = build_similarity_features(similarity_start)

    # position * 2, user * 2, CTR * 6, similarity * 2, id * (6(unknown) + lens)
    n = 2 + 2 + 6 + 2 + 6 + len(adIDs) + len(aderIDs) + len(queryIDs) + len(keywordIDs) + len(titleIDs) + len(userIDs)
    print n

    # coordinate sparse matrix
    m = file_len(file_read.name)
    # file_write.write("%%MatrixMarket matrix coordinate integer general" + '\n' + "%" +'\n');  # mm sparse matrix
    file_write.write(str(m) + " " + str(n) + " " + str(m*18) + '\n')   # row, column, number of values

    row = 0
    for line in file_read:
        features = {}
        fields = line.strip('\n').split('\t')

        # position [position, relative position]
        features[0] = fields[6]
        features[1] = round((float(fields[5]) - float(fields[6])) / float(fields[5]), 5)

        # user [age, gender]
        features[2] = user_profile[int(fields[11])][0]      # list indices must be integers, not str
        features[3] = user_profile[int(fields[11])][1]

        # CTR [ad, advertiser, query, keyword, title, user]
        features[4] = ctr_ad.setdefault(fields[3], 0.05)
        features[5] = ctr_ader.setdefault(fields[4], 0.05)
        features[6] = ctr_query.setdefault(fields[7], 0.05)
        features[7] = ctr_keyword.setdefault(fields[8], 0.05)
        features[8] = ctr_title.setdefault(fields[9], 0.05)
        features[9] = ctr_user.setdefault(fields[11], 0.05)

        # similarity, query-title, query-description
        features[10] = query_title_similarity[row]
        features[11] = query_desc_similarity[row]

        # ID [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]
        # ID类特征第一位留给unknown,所有整体后移一位
        offset = 12
        try:
            features[offset + adIDs[fields[3]] + 1] = 1     # 使用setdefault会改变矩阵的大小
        except IndexError:                                  # 不要使用value.key in dict.keys()，这样会新建一个key的list,
            features[offset] = 1                            # 可以用value.key in dict
        offset += (len(adIDs) + 1)

        try:
            features[offset + aderIDs[fields[4]] + 1] = 1
        except KeyError:
            features[offset] = 1
        offset += (len(aderIDs) + 1)

        try:
            features[offset + queryIDs[fields[7]] + 1] = 1
        except KeyError:
            features[offset] = 1
        offset += (len(queryIDs) + 1)

        try:
            features[offset + keywordIDs[fields[8]] + 1] = 1
        except KeyError:
            features[offset] = 1
        offset += (len(keywordIDs) + 1)

        try:
            features[offset + titleIDs[fields[9]] + 1] = 1
        except KeyError:
            features[offset] = 1
        offset += (len(titleIDs) + 1)

        try:
            features[offset + userIDs[fields[11]] + 1] = 1
        except KeyError:
            features[offset] = 1

        if int(fields[0]) > 0:
            fields[0] = '1'

        # file_write.write(fields[0]+' ')   # write y  libfm
        for col in features.keys():   # row and column of matrix market start from 1, coo matrix start from 0
            file_write.write(str(row) + " " + str(col) + " " + str(features[col]) + '\n')
            # file_write.write(str(col) + ":" + str(features[col]) + ' ')    # libfm
        # file_write.write('\n') # libfm

        if row % 500000 == 0:
            print row
        row += 1
        del features
    file_read.close()
    file_write.close()


def build_user_profile():
    # make user raw data
    user_profile_file = open(Constants.dir_path + "userid_profile.txt")
    user_profile = [['0', '0']]
    for line in user_profile_file:
        fields = line.strip('\n').split('\t')
        user_profile.append([fields[1], fields[2]])
    print "Buliding user profile finished."
    return user_profile


def build_x():
    stat = pd.read_csv(Constants.dir_path + "sample\\total.part", header=None, delimiter='\t', dtype=str)
    print "Reading file finished."

    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = build_id_features(stat)
    idset = [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]   # shallow copy, idset[0]和adIDs指向同一地址
    ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user = build_ctr(idset)
    ctr_set = [ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user]
    user_profile = build_user_profile()

    # data file definition
    train_from = open(Constants.dir_path + "sample\\training.part")
    train_to = open(Constants.dir_path + "sample\\embedding\\training.X4.embedding", "w")
    valid_from = open(Constants.dir_path + "sample\\validation.part")
    valid_to = open(Constants.dir_path + "sample\\embedding\\validation.X4.embedding", "w")
    test_from = open(Constants.dir_path + "sample\\test.part")
    test_to = open(Constants.dir_path + "sample\\embedding\\test.X4.embedding", "w")

    build_x_helper(idset, ctr_set, user_profile,  train_from, train_to, 0)
    build_x_helper(idset, ctr_set, user_profile,  valid_from, valid_to, 1800000)
    build_x_helper(idset, ctr_set, user_profile, test_from, test_to, 2000000)



if __name__ == '__main__':

    build_x()
    # stat = pd.read_csv(dir_path + "sample\\total.part", header=None, delimiter='\t', dtype=str)
    # print "read file finished."
    # adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = build_id(stat)
    # print "build id finished."
    # idset = [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]   #shallow copy, idset[0]和adIDs指向同一地址
    # ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user = build_ctr(stat, idset)
    # print ctr_ad[:20]
    # print ctr_ader[:20]
    # print ctr_query[:20]
    # print ctr_keyword[:20]
    # print ctr_title[:20]
    # print ctr_user[:20]
    # print stat[9].value_counts().head(10)