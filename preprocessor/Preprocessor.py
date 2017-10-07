# -*- coding:utf-8 -*-

import pandas as pd
from Utils import file_len
import constants
import id_handler
import ctr_handler
import gbdt_handler
import fm_handler
import user_handler
import io
import operator
import gc
from itertools import izip, starmap
import math

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
# [1] 将相似度特征拆为维度为10的离散特征. [2] 直接使用连续值
def build_similarity_features(start, discrete=False):
    simi_feature_file = open(constants.dir_path + "sample\\mapping\\txtCosDistance_clean.feature")
    query_title_simi = []
    query_desc_simi = []
    for line in simi_feature_file:
        tuple2 = line.strip('\n').split('\t')
        simi1 = float(tuple2[0])
        simi2 = float(tuple2[1])
        if discrete:
            query_title_simi.append(int(abs(simi1-0.0001) * 10))
            query_desc_simi.append(int(abs(simi2-0.0001) * 10))
        else:
            query_title_simi.append(simi1)
            query_desc_simi.append(simi2)
    print "start: " + str(start)
    print "similarity:" + str(query_title_simi[start]), str(query_desc_simi[start])
    return query_title_simi[start:], query_desc_simi[start:]


# 以coordinate稀疏矩阵存储
def write_as_coor(features, file_write, row):
    for col in features.keys():  # row and column of matrix market start from 1, coo matrix start from 0
        file_write.write(str(row) + " " + str(col) + " " + str(features[col]) + '\n')


# 以libfm形式存储
def write_as_libfm(features, file_write, fields):
    file_write.write(unicode(fields[0]+' '))   # write y  libfm
    for col in features.keys():  # row and column of matrix market start from 1, coo matrix start from 0
        file_write.write(unicode(str(col) + ":" + str(features[col]) + ' '))
    file_write.write(unicode('\n'))


# 以scipy稀疏矩阵形式存储, similarity_start标记相似度特征的起始位置
def build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, fm_v, file_read, file_write, similarity_start,
                   has_id=True,
                   has_gbdt=False,
                   has_fm=False,
                   file_format="fm",
                   dataset="train"):
    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = idset[0], idset[1], idset[2], idset[3], idset[4]\
        , idset[5]

    discrete = False
    query_title_similarity, query_desc_similarity = build_similarity_features(similarity_start, discrete)

    ''' position * 2, user * 2, (impression, click, CTR) * 8 * 3, combine ctr * 3, similarity * (2*(10)), fm * C(2, 20), GBDT * 600, id * (6(unknown) + lens)'''

    values = 2 + 2 + 18 + 3 + 2
    dim = 2 + 2 + 18 + 3 + 2

    if discrete:
        dim += 33

    if has_gbdt:
        values += 30
        dim += 600
    if has_fm:
        values += 190
        dim += 190
    if has_id:
        values += 6
        dim += 6 + len(adIDs) + len(aderIDs) + len(queryIDs) + len(keywordIDs) + len(titleIDs) + len(userIDs)
    print "dimension:" + str(dim)
    print "features:" + str(values)

    m = file_len(file_read.name)
    # file_write.write("%%MatrixMarket matrix coordinate integer general" + '\n' + "%" +'\n');  # mm sparse matrix

    if file_format == "coordinate":
        file_write.write(str(m) + " " + str(dim) + " " + str(m * values) + '\n')  # row, column, number of values

    if has_gbdt:
        gbdt_feature = gbdt_handler.build_gbdt(dataset)
        print "Building gbdt features finished"

    row = 0
    for line in file_read:
        features = {}
        fields = line.strip('\n').split('\t')

        # position [position, relative position]
        offset = 0
        features[0] = fields[6]
        features[1] = round((float(fields[5]) - float(fields[6])) / float(fields[5]), 5)

        # user [age, gender]
        features[2] = user_profile.setdefault(fields[11], ['0', '0'])[0]      # list indices must be integers, not str
        features[3] = user_profile.setdefault(fields[11], ['0', '0'])[1]
        offset = 4

        # CTR [ad, advertiser, depth, pos, query, keyword, title, user]
        for i in range(len(ctr_set)-1):
            features[offset+i] = ctr_set[i].setdefault(fields[i+3], 0.05)
        user_id_idx = len(ctr_set)-1     # user_id ctr is processed separately
        features[offset+user_id_idx] = ctr_set[user_id_idx].setdefault(fields[11], 0.05)
        offset += len(ctr_set)

        # impression number
        for i in range(len(impre_set)-1):
            features[offset+i] = round(math.log(impre_set[i].setdefault(fields[i+3], 1)+1, 10), 5)
        user_id_idx = len(impre_set)-1     # user_id ctr is processed separately
        features[offset+user_id_idx] = round(math.log(impre_set[user_id_idx].setdefault(fields[11], 1)+1, 10), 5)
        offset += len(impre_set)

        # click number
        for i in range(len(click_set)-1):
            features[offset+i] = round(math.log(click_set[i].setdefault(fields[i+3], 1)+1, 2), 5)
        user_id_idx = len(click_set)-1     # user_id ctr is processed separately
        features[offset+user_id_idx] = round(math.log(click_set[user_id_idx].setdefault(fields[11], 1)+1, 2), 5)
        offset += len(click_set)

        # combine ctr
        hash_ids = [hash(fields[3] + '_' + fields[7]) % 1e6, hash(fields[3] + '_' + fields[6]) % 1e6, hash(fields[3] + '_' + fields[11]) % 1e6]
        for i in range(len(combine_ctr_set)):
            features[offset+i] = combine_ctr_set[i].setdefault(hash_ids[i], 0.05)
        offset += len(combine_ctr_set)


        # features[4] = ctr_set[.setdefault(fields[3], 0.05)
        # features[5] = ctr_ader.setdefault(fields[4], 0.05)
        # features[6] = ctr_query.setdefault(fields[7], 0.05)
        # features[7] = ctr_keyword.setdefault(fields[8], 0.05)
        # features[8] = ctr_title.setdefault(fields[9], 0.05)
        # features[9] = ctr_user.setdefault(fields[11], 0.05)

        # similarity, query-title, query-description
        if discrete:
            features[offset+10+query_title_similarity[row]] = 1
            features[offset+20+query_desc_similarity[row]] = 1
            offset += 30
        else:
            features[offset+10] = query_title_similarity[row]
            features[offset+11] = query_desc_similarity[row]
            offset += 12

        if has_fm:
            k = 0
            for i in range(offset):
                for j in range(i+1, offset):
                    if i in features and j in features and features[j] != 0 and features[i] != 0:
                        features[k+offset] = float(features[i])*float(features[j])*sum(starmap(operator.mul, izip(fm_v[i], fm_v[j])))
                    k += 1
            offset += k

        if has_gbdt:
            for k in gbdt_feature[row]:
                features[k+offset] = 1
            offset += 600

        if has_id:
            # ID [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]
            # ID类特征第一位留给unknown,所有整体后移一位
            if fields[3] in adIDs:
                features[offset + adIDs[fields[3]] + 1] = 1     # 使用setdefault会改变矩阵的大小
            else:                                 # 不要使用value.key in dict.keys()，这样会新建一个key的list,
                features[offset] = 1                            # 可以用value.key in dict
            offset += (len(adIDs) + 1)

            if fields[4] in aderIDs:
                features[offset + aderIDs[fields[4]] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(aderIDs) + 1)

            if fields[7] in queryIDs:
                features[offset + queryIDs[fields[7]] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(queryIDs) + 1)

            if fields[8] in keywordIDs:
                features[offset + keywordIDs[fields[8]] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(keywordIDs) + 1)

            if fields[9] in titleIDs:
                features[offset + titleIDs[fields[9]] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(titleIDs) + 1)

            if fields[11] in userIDs:
                features[offset + userIDs[fields[11]] + 1] = 1
            else:
                features[offset] = 1

        if int(fields[0]) > 0:
            fields[0] = '1'

        if file_format == "coordinate":
            write_as_coor(features, file_write, row)
        else:
            write_as_libfm(features, file_write, fields)

        if row % 500000 == 0:
            print "rows: " + str(row)
        row += 1
        del features, fields
    file_read.close()
    file_write.close()


# 生成不进行onehot处理的特征
def build_x_no_transform(idset, ctr_set, user_profile, fm_v, file_read, file_write, similarity_start,
                   has_id=True,
                   has_gbdt=False,
                   has_fm=False,
                   dataset="train"):
    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = idset[0], idset[1], idset[2], idset[3], idset[4] \
        , idset[5]
    ctr_ad, ctr_ader, ctr_query, ctr_keyword, ctr_title, ctr_user = ctr_set[0], ctr_set[1], ctr_set[2], ctr_set[3] \
        , ctr_set[4], ctr_set[5]

    query_title_similarity, query_desc_similarity = build_similarity_features(similarity_start)

    ''' position * 2, user * 2, CTR * 6, similarity * 1, fm * C(2, 20), GBDT * 600, id * 6'''

    if has_gbdt:
        gbdt_feature = gbdt_handler.build_gbdt(dataset)
        print "Building gbdt features finished"

    row = 0
    for line in file_read:
        features = {}
        fields = line.strip('\n').split('\t')

        # position [position, relative position]
        features[0] = fields[6]
        features[1] = round((float(fields[5]) - float(fields[6])) / float(fields[5]), 5)

        # user [age, gender]
        features[2] = user_profile[int(fields[11])][0]  # list indices must be integers, not str
        features[3] = user_profile[int(fields[11])][1]

        # CTR [ad, advertiser, query, keyword, title, user]
        features[4] = ctr_ad.setdefault(fields[3], 0.05)
        features[5] = ctr_ader.setdefault(fields[4], 0.05)
        features[6] = ctr_query.setdefault(fields[7], 0.05)
        features[7] = ctr_keyword.setdefault(fields[8], 0.05)
        features[8] = ctr_title.setdefault(fields[9], 0.05)
        features[9] = ctr_user.setdefault(fields[11], 0.05)

        # similarity, query-title, query-description, 暂时丢弃description
        features[10] = query_title_similarity[row]
        # features[20+query_desc_similarity[row]] = 1

        offset = 11

        if has_fm:
            k = 0
            for i in range(offset):
                for j in range(i + 1, offset):
                    if i in features and j in features and features[j] != 0 and features[i] != 0:
                        features[k + offset] = float(features[i]) * float(features[j]) * sum(
                            starmap(operator.mul, izip(fm_v[i], fm_v[j])))
                    k += 1
            offset += k

        if has_gbdt:
            for k in gbdt_feature[row]:
                features[k + offset] = 1
            offset += 600

        if has_id:
            # ID [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]
            # ID类特征第一位留给unknown,所有整体后移一位
            if fields[3] in adIDs:
                features[offset] = adIDs[fields[3]] + 1  # 使用setdefault会改变矩阵的大小
            else:  # 不要使用value.key in dict.keys()，这样会新建一个key的list,
                features[offset] = 0  # 可以用value.key in dict
            offset += 1

            if fields[4] in aderIDs:
                features[offset] = aderIDs[fields[4]] + 1
            else:
                features[offset] = 0
            offset += 1

            if fields[7] in queryIDs:
                features[offset] = queryIDs[fields[7]] + 1
            else:
                features[offset] = 0
            offset += 1

            if fields[8] in keywordIDs:
                features[offset] = keywordIDs[fields[8]] + 1
            else:
                features[offset] = 0
            offset += 1

            if fields[9] in titleIDs:
                features[offset] = titleIDs[fields[9]] + 1
            else:
                features[offset] = 0
            offset += 1

            if fields[11] in userIDs:
                features[offset] = userIDs[fields[11]] + 1
            else:
                features[offset] = 0

        if int(fields[0]) > 0:
            fields[0] = '1'

        write_as_libfm(features, file_write, fields)

        if row % 500000 == 0:
            print "rows: " + str(row)
        row += 1
        del features
    file_read.close()
    file_write.close()



def build_x():
    stat = pd.read_csv(constants.dir_path + "sample\\total.part", header=None, delimiter='\t', dtype=str)
    print "Reading file finished."

    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = id_handler.build_id_features(stat)
    idset = [adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs]   # shallow copy, idset[0]和adIDs指向同一地址
    impre_set, click_set, ctr_set, combine_ctr_set = ctr_handler.build_ctr(idset)
    user_profile = user_handler.build_user_profile(userIDs)
    fm_v = fm_handler.build_fm_features()

    # data file definition, newline is necessary when write as libfm format
    train_from = open(constants.dir_path + "sample\\training.part")
    train_to = open(constants.dir_path + "sample\\features\\train.new_basic.libfm", "w")
    valid_from = open(constants.dir_path + "sample\\validation.part")
    valid_to = open(constants.dir_path + "sample\\features\\validation.new_basic.libfm", "w")
    test_from = open(constants.dir_path + "sample\\test.part")
    test_to = open(constants.dir_path + "sample\\features\\test.new_basic.libfm", "w")

    build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, fm_v, train_from, train_to, 0,
                   has_id=True, has_fm=False, file_format="fm", dataset="train")
    build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, fm_v, valid_from, valid_to, 1800000,
                   has_id=True, has_fm=False, file_format="fm", dataset="validation")
    build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, fm_v, test_from, test_to, 2000000,
                   has_id=True, has_fm=False, file_format="fm", dataset="test")

    # build_x_no_transform(idset, ctr_set, user_profile, fm_v, train_from, train_to, 0,
    #                has_id=True, has_fm=False, dataset="train")
    # build_x_no_transform(idset, ctr_set, user_profile, fm_v, valid_from, valid_to, 1800000,
    #                has_id=True, has_fm=False, dataset="validation")
    # build_x_no_transform(idset, ctr_set, user_profile, fm_v, test_from, test_to, 2000000,
    #                has_id=True, has_fm=False,  dataset="test")


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