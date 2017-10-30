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
from sklearn import preprocessing, datasets
import numpy as np


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
    for col in sorted(features.keys()):  # row and column of matrix market start from 1, coo matrix start from 0
        file_write.write(unicode(str(col) + ":" + str(features[col]) + ' '))
    file_write.write(unicode('\n'))


# 以scipy稀疏矩阵形式存储, similarity_start标记相似度特征的起始位置
def build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, file_read, file_write, similarity_start,
                   fm_v=None,
                   has_id=True,
                   has_gbdt=False,
                   has_fm=False,
                   file_format="fm",
                   dataset="train",
                   has_user=False):
    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = idset[0], idset[1], idset[4], idset[5], idset[6]\
        , idset[8]

    discrete = False
    query_title_similarity, query_desc_similarity = build_similarity_features(similarity_start, discrete)

    ''' position * 2, depth * 1, user * 2, (impression, click, CTR) * 8 * 3, combine ctr * 3, similarity * (2*(10)), 
    fm * C(2, 20), GBDT * 600, id * (6(unknown) + lens)'''

    values = 2 + 1 + 2 + 8 + 2
    dim = 5 + 4 + 10 + 8 + 2

    if discrete:
        dim += 18
    if has_gbdt:
        values += 30
        dim += 600
    if has_fm:
        values += 190
        dim += 190
    if has_id:
        values += 6
        dim += 6 + len(adIDs) + len(aderIDs) + len(queryIDs) + len(keywordIDs) + len(titleIDs)
        if has_user:
            dim += + len(userIDs)
    #
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

        adId, adverId, depth, pos, queryId, keywordId, titleId, desId, userId = fields[3], fields[4], fields[5], \
                                                                                fields[6], fields[7], fields[8], \
                                                                                fields[9], fields[10], fields[11]

        # position [position, relative position]
        offset = 0
        features[offset+int(pos)] = 1
        offset += 4
        features[offset] = round((float(fields[5]) - float(fields[6])) / float(fields[5]), 5)
        offset += 1

        # depth
        features[offset+int(depth)] = 1
        offset += 4
        # user [age, gender]
        age = int(user_profile.setdefault(userId, ['0', '0'])[1])
        features[offset+age] = 1
        offset += 7
        gender = int(user_profile.setdefault(userId, ['0', '0'])[0])
        features[offset+gender] = 1
        offset += 3

        # CTR [ad, advertiser, depth, pos, query, keyword, title, description, user]

        ctr_ad, ctr_adver, ctr_depth, ctr_pos, ctr_query, ctr_keyword, ctr_title, ctr_user = ctr_set[0], ctr_set[1], \
                                                                                             ctr_set[2], ctr_set[3], \
                                                                                             ctr_set[4], ctr_set[5], \
                                                                                             ctr_set[6], ctr_set[8]

        impre_ad, impre_adver, impre_depth, impre_pos, impre_query, impre_keyword, impre_title, impre_user = impre_set[0], impre_set[1], \
                                                                                                             impre_set[2], impre_set[3], \
                                                                                                             impre_set[4], impre_set[5], \
                                                                                                             impre_set[6], impre_set[8]

        # for i in range(len(ctr_set)):
        #     features[offset+i] = ctr_set[i].setdefault(fields[i+3], 0.05)
        # offset += len(ctr_set)

        features[offset + 0] = ctr_ad.setdefault(adId, 0.05)
        features[offset + 1] = ctr_adver.setdefault(adverId, 0.05)
        features[offset + 2] = ctr_pos.setdefault(pos, 0.05)
        features[offset + 3] = ctr_query.setdefault(queryId, 0.05)
        features[offset + 4] = ctr_keyword.setdefault(keywordId, 0.05)
        features[offset + 5] = ctr_title.setdefault(titleId, 0.05)
        features[offset + 6] = ctr_user.setdefault(userId, 0.05)
        features[offset + 7] = ctr_depth.setdefault(depth, 0.05)
        offset += 8


        # impression number
        # for i in range(len(impre_set)):
        #     features[offset+i] = impre_set[i].setdefault(fields[i+3], 0)
        # offset += len(impre_set)

        # features[offset + 1] = round(math.log(impre_ad.setdefault(adId, 0)+1, 10), 4)
        # features[offset + 2] = round(math.log(impre_adver.setdefault(adverId, 0)+1, 10), 4)
        # features[offset + 4] = round(math.log(impre_query.setdefault(queryId, 0)+1, 10), 4)
        # features[offset + 5] = round(math.log(impre_keyword.setdefault(keywordId, 0)+1, 10), 4)
        # features[offset + 6] = round(math.log(impre_title.setdefault(titleId, 0)+1, 10), 4)
        # features[offset + 7] = round(math.log(impre_user.setdefault(userId, 0)+1, 10), 4)
        # offset += 8


        # click number
        # for i in range(len(click_set)):
        #     features[offset+i] = click_set[i].setdefault(fields[i+3], 0)
        # offset += len(click_set)

        # combine ctr
        # hash_ids = [hash(fields[3] + '_' + fields[7]) % 1e6, hash(fields[3] + '_' + fields[6]) % 1e6, hash(fields[3] + '_' + fields[11]) % 1e6]
        # for i in range(len(combine_ctr_set)):
        #     features[offset+i] = combine_ctr_set[i].setdefault(hash_ids[i], 0.05)
        # offset += len(combine_ctr_set)

        # similarity, query-title, query-description
        if discrete:
            features[offset+query_title_similarity[row]] = 1
            features[offset+10+query_desc_similarity[row]] = 1
            offset += 20
        else:
            features[offset] = query_title_similarity[row]
            features[offset+1] = query_desc_similarity[row]
            offset += 2

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
            if adId in adIDs:
                features[offset + adIDs[adId] + 1] = 1     # 使用setdefault会改变矩阵的大小
            else:                                 # 不要使用value.key in dict.keys()，这样会新建一个key的list,
                features[offset] = 1                            # 可以用value.key in dict
            offset += (len(adIDs) + 1)

            if adverId in aderIDs:
                features[offset + aderIDs[adverId] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(aderIDs) + 1)

            if queryId in queryIDs:
                features[offset + queryIDs[queryId] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(queryIDs) + 1)

            if keywordId in keywordIDs:
                features[offset + keywordIDs[keywordId] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(keywordIDs) + 1)

            if titleId in titleIDs:
                features[offset + titleIDs[titleId] + 1] = 1
            else:
                features[offset] = 1
            offset += (len(titleIDs) + 1)

            if has_user:
                if userId in userIDs:
                    features[offset + userIDs[userId] + 1] = 1
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
def build_x_no_transform(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, file_read, file_write, similarity_start,
                   fm_v=None,
                   has_id=True,
                   has_gbdt=False,
                   has_fm=False,
                   file_format="fm",
                   dataset="train"):
    adIDs, aderIDs, queryIDs, keywordIDs, titleIDs, userIDs = idset[0], idset[1], idset[4], idset[5], idset[6]\
        , idset[8]

    discrete = False
    query_title_similarity, query_desc_similarity = build_similarity_features(similarity_start, discrete)

    ''' position * 2, user * 2, (impression, click, CTR) * 9 * 3, combine ctr * 3, similarity * (2*(10)), fm * C(2, 20), GBDT * 600, id * (6(unknown) + lens)'''

    values = 2 + 2 + 27 + 3 + 2
    dim = 2 + 2 + 27 + 3 + 2

    if discrete:
        dim += 18
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

    if has_gbdt:
        gbdt_feature = gbdt_handler.build_gbdt(dataset)
        print "Building gbdt features finished"

    row = 0
    for line in file_read:
        features = {}
        fields = line.strip('\n').split('\t')

        adId, adverId, depth, pos, queryId, keywordId, titleId, desId, userId = fields[3], fields[4], fields[5], \
                                                                                fields[6], fields[7], fields[8], \
                                                                                fields[9], fields[10], fields[11]

        # position [position, relative position]
        offset = 0
        features[offset] = int(pos)
        offset += 1
        features[offset] = round((float(fields[5]) - float(fields[6])) / float(fields[5]), 5)
        offset += 1
        features[offset] = int(depth)
        offset += 1

        # user [age, gender]
        age = int(user_profile.setdefault(fields[11], ['0', '0'])[0])
        features[offset] = age
        offset += 1
        gender = int(user_profile.setdefault(fields[11], ['0', '0'])[1])
        features[offset] = gender
        offset += 1

        # CTR [ad, advertiser, depth, pos, query, keyword, title, description, user]

        ctr_ad, ctr_adver, ctr_depth, ctr_pos, ctr_query, ctr_keyword, ctr_title, ctr_user = ctr_set[0], ctr_set[1], \
                                                                                             ctr_set[2], ctr_set[3], \
                                                                                             ctr_set[4], ctr_set[5], \
                                                                                             ctr_set[6], ctr_set[8]

        impre_ad, impre_adver, impre_depth, impre_pos, impre_query, impre_keyword, impre_title, impre_user = impre_set[0], impre_set[1], \
                                                                                                             impre_set[2], impre_set[3], \
                                                                                                             impre_set[4], impre_set[5], \
                                                                                                             impre_set[6], impre_set[8]

        # for i in range(len(ctr_set)):
        #     features[offset+i] = ctr_set[i].setdefault(fields[i+3], 0.05)
        # offset += len(ctr_set)

        features[offset + 1] = ctr_ad.setdefault(adId, 0.05)
        features[offset + 2] = ctr_adver.setdefault(adverId, 0.05)
        features[offset + 3] = ctr_pos.setdefault(pos, 0.05)
        features[offset + 4] = ctr_query.setdefault(queryId, 0.05)
        features[offset + 5] = ctr_keyword.setdefault(keywordId, 0.05)
        features[offset + 6] = ctr_title.setdefault(titleId, 0.05)
        features[offset + 7] = ctr_user.setdefault(userId, 0.05)
        features[offset + 8] = ctr_depth.setdefault(depth, 0.05)
        offset += 10


        # impression number
        # for i in range(len(impre_set)):
        #     features[offset+i] = impre_set[i].setdefault(fields[i+3], 0)
        # offset += len(impre_set)

        # features[offset + 1] = round(math.log(impre_ad.setdefault(adId, 0)+1, 10), 4)
        # features[offset + 2] = round(math.log(impre_adver.setdefault(adverId, 0)+1, 10), 4)
        # features[offset + 4] = round(math.log(impre_query.setdefault(queryId, 0)+1, 10), 4)
        # features[offset + 5] = round(math.log(impre_keyword.setdefault(keywordId, 0)+1, 10), 4)
        # features[offset + 6] = round(math.log(impre_title.setdefault(titleId, 0)+1, 10), 4)
        # features[offset + 7] = round(math.log(impre_user.setdefault(userId, 0)+1, 10), 4)
        # offset += 8


        # click number
        # for i in range(len(click_set)):
        #     features[offset+i] = click_set[i].setdefault(fields[i+3], 0)
        # offset += len(click_set)

        # combine ctr
        # hash_ids = [hash(fields[3] + '_' + fields[7]) % 1e6, hash(fields[3] + '_' + fields[6]) % 1e6, hash(fields[3] + '_' + fields[11]) % 1e6]
        # for i in range(len(combine_ctr_set)):
        #     features[offset+i] = combine_ctr_set[i].setdefault(hash_ids[i], 0.05)
        # offset += len(combine_ctr_set)

        # similarity, query-title, query-description
        if discrete:
            features[offset+query_title_similarity[row]] = 1
            features[offset+10+query_desc_similarity[row]] = 1
            offset += 20
        else:
            features[offset] = query_title_similarity[row]
            features[offset+1] = query_desc_similarity[row]
            offset += 2

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
            if adId in adIDs:
                features[offset] = adIDs[adId]     # 使用setdefault会改变矩阵的大小
            else:                                 # 不要使用value.key in dict.keys()，这样会新建一个key的list,
                features[offset] = 0                            # 可以用value.key in dict
            offset += 1

            if adverId in aderIDs:
                features[offset] = aderIDs[adverId]
            else:
                features[offset] = 0
            offset += 1

            if queryId in queryIDs:
                features[offset] = queryIDs[queryId]
            else:
                features[offset] = 0
            offset += 1

            if keywordId in keywordIDs:
                features[offset] = keywordIDs[keywordId]
            else:
                features[offset] = 0
            offset += 1

            if titleId in titleIDs:
                features[offset] = titleIDs[titleId]
            else:
                features[offset] = 0
            offset += 1

            if userId in userIDs:
                features[offset] = userIDs[userId]
            else:
                features[offset] = 0

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


def build_x():

    train_to_path = constants.dir_path + "sample\\features\\train.nn_no-cl-im-comb_t15.libfm"
    valid_to_path = constants.dir_path + "sample\\features\\valid.nn_no-cl-im-comb_t15.libfm"
    test_to_path = constants.dir_path + "sample\\features\\test.nn_no-cl-im-comb_t15.libfm"

    stat = pd.read_csv(constants.dir_path + "sample\\total.part", header=None, delimiter='\t', dtype=str)
    print "Reading file finished."

    adIDs, aderIDs, depthIDs, posIDs, queryIDs, keywordIDs, titleIDs, desIDs, userIDs = id_handler.build_id_features(stat, 50)
    idset = [adIDs, aderIDs, depthIDs, posIDs, queryIDs, keywordIDs, titleIDs, desIDs, userIDs]   # shallow copy, idset[0]和adIDs指向同一地址
    impre_set, click_set, ctr_set, combine_ctr_set = ctr_handler.build_ctr(idset)
    user_profile = user_handler.build_user_profile(userIDs)
    # fm_v = fm_handler.build_fm_features()


    # data file definition, newline is necessary when write as libfm format
    train_from = open(constants.dir_path + "sample\\training.part")
    train_to = io.open(train_to_path, "w", newline='\n')
    # valid_from = open(constants.dir_path + "sample\\validation.part")
    # valid_to = io.open(valid_to_path, "w", newline='\n')
    test_from = open(constants.dir_path + "sample\\test.part")
    test_to = io.open(test_to_path, "w", newline='\n')

    build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, train_from, train_to, 0,
                   fm_v=None, has_id=True, has_fm=False, file_format="fm", dataset="train", has_user=True)
    # build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, valid_from, valid_to, 1800000,
    #                fm_v=None, has_id=True, has_fm=False, file_format="fm", dataset="validation", has_user=False)
    build_x_helper(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, test_from, test_to, 2000000,
                   fm_v=None, has_id=True, has_fm=False, file_format="fm", dataset="test", has_user=True)

    # build_x_no_transform(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, train_from, train_to, 0,
    #                fm_v=None, has_id=True, has_fm=False, file_format="fm", dataset="train")
    # build_x_no_transform(idset, impre_set, click_set, ctr_set, combine_ctr_set, user_profile, test_from, test_to, 2000000,
    #                fm_v=None, has_id=True, has_fm=False, file_format="fm", dataset="test")


    # feature_files = [train_to_path, test_to_path]
    # scale_x(feature_files)
    # return


# scale data to [0, 1]
def scale_x(feature_files):
    min_max_scaler = preprocessing.MaxAbsScaler()
    for f in feature_files:
        x, y = datasets.load_svmlight_file(f)
        x_scale = np.round(min_max_scaler.fit_transform(x), 4)
        datasets.dump_svmlight_file(x_scale, y, f)
        print str(f) + " finished."




if __name__ == '__main__':
    build_x()
    # scale_x(feature_files)
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