# -*- coding:utf-8 -*-

import pandas as pd
from Utils import file_len
import constants
from id_handler import *
from ctr_handler import *

def build_Y():
    file_read = open(constants.dir_path + "sample\\test.part")
    file_Y = open(constants.dir_path + "sample\\test.Y", 'w')
    for line in file_read:
        y = line[:1]
        if int(y) > 0: y = 1
        file_Y.write(str(y) + '\n')
    file_read.close()
    file_Y.close()


# query-title, query-description, use start end to locate row of record
def build_similarity_features(start):
    simi_feature_file = open(constants.dir_path + "sample\\mapping\\txtNormDistance_2.feature")
    query_title_simi = []
    query_desc_simi = []
    for line in simi_feature_file:
        tuple2 = line.strip('\n').split('\t')
        query_title_simi.append(tuple2[0])
        query_desc_simi.append(tuple2[1])
    print "similarity:" + query_title_simi[start], query_desc_simi[start]
    return query_title_simi[start:], query_desc_simi[start:]


# 以coordinate稀疏矩阵存储
def write_as_coor(features, file_write, row):
    for col in features.keys():  # row and column of matrix market start from 1, coo matrix start from 0
        file_write.write(str(row) + " " + str(col) + " " + str(features[col]) + '\n')


# 以libfm形式存储
def write_as_libfm(features, file_write, fields):
    file_write.write(fields[0]+' ')   # write y  libfm
    for col in features.keys():  # row and column of matrix market start from 1, coo matrix start from 0
        file_write.write(str(col) + ":" + str(features[col]) + ' ')
    file_write.write('\n')


# 以scipy稀疏矩阵形式存储, similarity_start标记相似度特征的起始位置
def build_x_helper(idset, ctr_set, user_profile, file_read, file_write, similarity_start):
    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = idset[0], idset[1], idset[2], idset[3], idset[4] \
        , idset[5]
    ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user = ctr_set[0], ctr_set[1], ctr_set[2], ctr_set[3] \
        , ctr_set[4], ctr_set[5]

    query_title_similarity, query_desc_similarity = build_similarity_features(similarity_start)

    # position * 2, user * 2, CTR * 6, similarity * 2, id * (6(unknown) + lens)
    # n = 2 + 2 + 6 + 2 + 6 + len(adIDs) + len(aderIDs) + len(queryIDs) + len(keywordIDs) + len(titleIDs) + len(userIDs)
    n = 2 + 2 + 6 + 2 + 6
    print n

    m = file_len(file_read.name)
    # file_write.write("%%MatrixMarket matrix coordinate integer general" + '\n' + "%" +'\n');  # mm sparse matrix

    file_format = "coordinate"
    if file_format == "coordinate":
        file_write.write(str(m) + " " + str(n) + " " + str(m * 18) + '\n')  # row, column, number of values

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

        if False:
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

        if file_format == "coordinate":
            write_as_coor(features, file_write, row)
        else:
            write_as_libfm(features, file_write, fields)

        if row % 500000 == 0:
            print row
        row += 1
        del features
    file_read.close()
    file_write.close()


def build_user_profile():
    # make user raw data
    user_profile_file = open(constants.dir_path + "userid_profile.txt")
    user_profile = [['0', '0']]
    for line in user_profile_file:
        fields = line.strip('\n').split('\t')
        user_profile.append([fields[1], fields[2]])
    print "Buliding user profile finished."
    return user_profile


def build_x():
    stat = pd.read_csv(constants.dir_path + "sample\\total.part", header=None, delimiter='\t', dtype=str)
    print "Reading file finished."

    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = build_id_features(stat)
    idset = [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]   # shallow copy, idset[0]和adIDs指向同一地址
    ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user = build_ctr(idset)
    ctr_set = [ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user]
    user_profile = build_user_profile()

    # data file definition
    train_from = open(constants.dir_path + "sample\\training.part")
    train_to = open(constants.dir_path + "sample\\features\\training.coor", "w")
    valid_from = open(constants.dir_path + "sample\\validation.part")
    valid_to = open(constants.dir_path + "sample\\features\\validation.coor", "w")
    test_from = open(constants.dir_path + "sample\\test.part")
    test_to = open(constants.dir_path + "sample\\features\\test.coor", "w")

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