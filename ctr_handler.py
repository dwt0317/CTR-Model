
import constants

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

    stat_file = open(constants.dir_path + "sample\\total.part", 'r')
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