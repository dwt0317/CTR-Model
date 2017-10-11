# -*- coding:utf-8 -*-
import constants


def ctr_helper(idset, impre_set, click_set):
    alpha = 0.05  # for smoothing
    beta = 75
    ctr_set = []
    print len(idset), len(impre_set), len(click_set)
    for i in range(len(idset)):
        ids = idset[i]
        ctrs = {}
        for id in ids:
            impression = float(impre_set[i].setdefault(id, 0))
            click = float(click_set[i].setdefault(id, 0))
            ctr = (click + alpha * beta) / (impression + beta)
            ctrs[id] = round(ctr, 5)
        ctr_set.append(ctrs)
    print "Computing ctr finished."
    return ctr_set


def combine_ctr_helper(idset, impre_set, click_set):
    alpha = 0.05  # for smoothing
    beta = 75
    combine_ctr_set = []
    for i in range(len(idset)):
        ctrs = {}
        for id in idset[i]:
            impression = float(impre_set[i].setdefault(id, 0))
            click = float(click_set[i].setdefault(id, 0))
            ctr = (click + alpha * beta) / (impression + beta)
            ctrs[id] = round(ctr, 5)
        combine_ctr_set.append(ctrs)
    print "Computing combine ctr finished."
    return combine_ctr_set


# CTR features [ad, advertiser, depth, pos, query, keyword, title, description user], [ad_query, ad_position, ad_user]
def build_ctr(idset):
    impre_ad, impre_ader, impre_keyword, impre_user, impre_query, impre_title, impre_des = {}, {}, {}, {}, {}, {}, {}
    click_ad, click_ader, click_keyword, click_user, click_query, click_title, click_des = {}, {}, {}, {}, {}, {}, {}

    impre_depth, impre_pos = {}, {}
    click_depth, click_pos = {}, {}

    impre_ad_query, impre_ad_position, impre_ad_user = {}, {}, {}
    click_ad_query, click_ad_position, click_ad_user = {}, {}, {}

    ad_query_ids, ad_position_ids, ad_user_ids = set(), set(), set()
    stat_file = open(constants.dir_path + "sample\\training.part", 'r')
    for line in stat_file:            # 迭代pandas太慢了，不要用
        row = line.strip('\n').split('\t')

        ad_query_hash = hash(row[3] + '_' + row[7]) % 1e6
        ad_position_hash = hash(row[3] + '_' + row[6]) % 1e6
        ad_user_hash = hash(row[3] + '_' + row[11]) % 1e6

        ad_query_ids.add(ad_query_hash)
        ad_position_ids.add(ad_position_hash)
        ad_user_ids.add(ad_user_hash)

        impre_ad[row[3]] = 1 + impre_ad.setdefault(row[3], 0)
        impre_ader[row[4]] = 1 + impre_ader.setdefault(row[4], 0)
        impre_depth[row[5]] = 1 + impre_depth.setdefault(row[5], 0)
        impre_pos[row[6]] = 1 + impre_pos.setdefault(row[6], 0)
        impre_query[row[7]] = 1 + impre_query.setdefault(row[7], 0)
        impre_keyword[row[8]] = 1 + impre_keyword.setdefault(row[8], 0)
        impre_title[row[9]] = 1 + impre_title.setdefault(row[9], 0)
        impre_des[row[10]] = 1 + impre_des.setdefault(row[10], 0)
        impre_user[row[11]] = 1 + impre_user.setdefault(row[11], 0)

        impre_ad_query[ad_query_hash] = 1 + impre_ad_query.setdefault(ad_query_hash, 0)
        impre_ad_position[ad_position_hash] = 1 + impre_ad_position.setdefault(ad_position_hash, 0)
        impre_ad_user[ad_user_hash] = 1 + impre_ad_user.setdefault(ad_user_hash, 0)

        if int(row[0]) == 1:
            click_ad[row[3]] = 1 + click_ad.setdefault(row[3], 0)
            click_ader[row[4]] = 1 + click_ader.setdefault(row[4], 0)
            click_depth[row[5]] = 1 + click_depth.setdefault(row[5], 0)
            click_pos[row[6]] = 1 + click_pos.setdefault(row[6], 0)
            click_query[row[7]] = 1 + click_query.setdefault(row[7], 0)
            click_keyword[row[8]] = 1 + click_keyword.setdefault(row[8], 0)
            click_title[row[9]] = 1 + click_title.setdefault(row[9], 0)
            click_des[row[10]] = 1 + click_des.setdefault(row[10], 0)
            click_user[row[11]] = 1 + click_user.setdefault(row[11], 0)

            click_ad_query[ad_query_hash] = 1 + click_ad_query.setdefault(ad_query_hash, 0)
            click_ad_position[ad_position_hash] = 1 + click_ad_position.setdefault(ad_position_hash, 0)
            click_ad_user[ad_user_hash] = 1 + click_ad_user.setdefault(ad_user_hash, 0)

    print "Counting impression and click finished."
    impre_set = [impre_ad, impre_ader, impre_depth, impre_pos, impre_query, impre_keyword, impre_title, impre_des, impre_user]
    click_set = [click_ad, click_ader, click_depth, click_pos, click_query, click_keyword, click_title, click_des, click_user]

    combine_impre_set = [impre_ad_query, impre_ad_position, impre_ad_user]
    combine_click_set = [click_ad_query, click_ad_position, click_ad_user]
    combine_id_set = [ad_query_ids, ad_position_ids, ad_user_ids]
    combine_ctr_set = combine_ctr_helper(combine_id_set, combine_impre_set, combine_click_set)
    del combine_id_set

    return impre_set, click_set, ctr_helper(idset, impre_set, click_set), combine_ctr_set
