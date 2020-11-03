from riotwatcher import LolWatcher, ApiError, TftWatcher
from utils.utils import print_if_complete
import pandas as pd
import json
import collections
from scipy.stats import skew, tstd, tmean
from time import sleep
from progress.bar import Bar
from multiprocessing.dummy import Pool as ThreadPool

# Global variables
api_key = 'RGAPI-90b27c39-5cdc-4a06-84a9-25b77a6b225d'
watcher = LolWatcher(api_key)
my_region = 'kr'
my_summoner_id = 'HardcoreZealot'


@print_if_complete
def league_id_lst(league):
    # Get user puuid list in certain league
    def single_league_id(summoner_info):
        id = summoner_info['summonerId']
        summoner = watcher.summoner.by_id(my_region, id)
        return summoner['accountId']

    pool = ThreadPool(3)
    id_lst = pool.map(single_league_id, league)

    return id_lst


@print_if_complete
def unique_match_id_lst(id_lst, count=10):
    # Get unqiue match id lst from a given list of user puuid
    res_lst = list()

    def unique_match_id_single(match):
        return match['gameId']

    for id in id_lst:
        try:
            match_lst = watcher.match.matchlist_by_account(
                my_region, id, '420', end_index=count)
        except:
            continue
        pool = ThreadPool(3)
        res_lst = res_lst + \
            pool.map(unique_match_id_single, match_lst['matches'])

    res_lst = list(set(res_lst))  # Remove Duplicates

    return res_lst


@print_if_complete
def match_info(match_id_lst):
    # Return Dataframe of match info from a given list of match ids
    info_lst = [["match_id", "team", "bot_support",
                 "bot_carry", "mid", "jungle", "top", "win"]]
    with Bar('Processing...') as bar:
        for match_id in match_id_lst:
            try:
                match = watcher.match.by_id(my_region, match_id)
            except:
                continue
            info_lst = append_match_info_lst(100, info_lst, match_id, match)
            info_lst = append_match_info_lst(200, info_lst, match_id, match)
            bar.next()

    match_df = pd.DataFrame(info_lst[1:], columns=info_lst[0])

    return match_df


def update_row_lst(temp_lst, index, accountId, not_assigned):
    if(temp_lst[index] != None):
        not_assigned.append(accountId)
    else:
        temp_lst[index] = accountId
    return not_assigned, temp_lst


def append_match_info_lst(team_id, info_lst, match_id, match):
    temp_lst = [None] * len(info_lst[0])
    temp_lst[0] = match_id
    temp_lst[1] = team_id
    if team_id == 100:
        temp_lst[7] = match["teams"][0]['win']
    elif team_id == 200:
        temp_lst[7] = match["teams"][1]['win']

    not_assigned = []
    for par_info, acc_info in zip(match["participants"], match['participantIdentities']):
        accountId = acc_info['player']['accountId']
        if par_info['teamId'] == team_id:
            if par_info['timeline']['lane'] == 'BOTTOM':
                if par_info['timeline']['role'] == "DUO_SUPPORT":
                    not_assigned, temp_lst = update_row_lst(
                        temp_lst, 2, accountId, not_assigned)
                else:
                    not_assigned, temp_lst = update_row_lst(
                        temp_lst, 3, accountId, not_assigned)
            elif par_info['timeline']['lane'] == 'MIDDLE':
                not_assigned, temp_lst = update_row_lst(
                    temp_lst, 4, accountId, not_assigned)
            elif par_info['timeline']['lane'] == 'JUNGLE':
                not_assigned, temp_lst = update_row_lst(
                    temp_lst, 5, accountId, not_assigned)
            elif par_info['timeline']['lane'] == 'TOP':
                not_assigned, temp_lst = update_row_lst(
                    temp_lst, 6, accountId, not_assigned)
            else:
                not_assigned.append(accountId)

    for index, elem in enumerate(temp_lst):
        if elem is None:
            temp_lst[index] = not_assigned[0]
            not_assigned.remove(temp_lst[index])

    info_lst.append(temp_lst)
    return info_lst


def player_info(accountId):

    try:
        summoner = watcher.summoner.by_account(my_region, accountId)
        summonerId = summoner['id']
        league = watcher.league.by_summoner(my_region, summonerId)[0]
    except:
        return None, None, None, None, None

    level = summoner['summonerLevel']
    total_win = league['wins']
    total_loss = league['losses']
    hot_streak = int(league['hotStreak'])

    data = [1] * total_win + [0] * total_loss
    win_skew = skew(data)
    win_std = tstd(data)
    win_mean = tmean(data)

    '''
    match_lst = watcher.match.matchlist_by_account(
        my_region, accountId, end_index=30, queue='420')
    for match in match_lst:
        print(match)
    '''

    return win_mean, win_std, win_skew, level, hot_streak


@print_if_complete
def feature_extraction(match_df):
    match_lst = [match_df.columns.values.tolist()] + match_df.values.tolist()

    new_col = []
    for name_index in range(2, 7):
        for extra in ['win_mean', 'win_std', 'win_skew', 'level', 'hot_streak']:
            new_col.append(str(match_lst[0][name_index]) + '_' + extra)

    for index, match_data in enumerate(match_lst[1:]):
        for role_index in range(2, 7):
            match_data = match_data + \
                list(player_info(match_data[role_index]))
        match_lst[index+1] = match_data
        match_df = pd.DataFrame(
            match_lst[1:index+1], columns=match_lst[0]+new_col)
        match_df = match_df.drop(["match_id", "bot_support",
                                  "bot_carry", "mid", "jungle", "top"], axis=1)
        match_df.to_csv('data/match_feature_test.csv', index=False)

    return match_df


def main():
    '''
    silver1_1 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 1)
    silver1_2 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 2)
    silver1_3 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 3)

    silver1_4 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 4)

    silver1_5 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 5)


    #silver1 = silver1_1 + silver1_2 + silver1_3
    silver1 = silver1_4
    '''
    gold1 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'GOLD', 'III', 1)

    id_lst = league_id_lst(gold1)
    with open("data/id_lst_gold.txt", "w") as fp:
        json.dump(id_lst, fp)

    with open("data/id_lst_gold.txt", "r") as fp:
        id_lst = json.load(fp)

    match_id_lst = unique_match_id_lst(id_lst)
    with open("data/match_id_lst_gold.txt", "w") as fp:
        json.dump(match_id_lst, fp)

    with open("data/match_id_lst_gold.txt", "r") as fp:
        match_id_lst = json.load(fp)
    '''
    match_df = match_info(match_id_lst)
    match_df.to_csv('data/match_gold.csv', index=False)

    match_df = pd.read_csv("data/match_gold.csv")
    match_df = feature_extraction(match_df)
    match_df.to_csv('data/match_feature_gold.csv', index=False)
    '''

@print_if_complete
def predict_feature(summonerId_lst, model):
    accountId_lst = [watcher.summoner.by_name(
        my_region, id)['accountId'] for id in summonerId_lst]

    match_data = [100]
    mean_mult = 1

    pool = ThreadPool(5)

    def temp_append(accountId):
        add = list(player_info(accountId))
        return add, add[0]

    results = pool.map(temp_append, accountId_lst)
    new = []
    for (x, y) in results:
        mean_mult = mean_mult * y
        new = new + x
    match_data = match_data + new

    match_data.append(mean_mult)
    # print(match_data)
    return model.predict([match_data])


if __name__ == "__main__":
    main()
    # me = watcher.summoner.by_id(my_region, my_summoner_id)
    # accountId = me['accountId']
    # player_info('pRzYZSsbfU4Ha0SlczOED7iT_mtDn-BuKvaLMZNNP1w8ZfEvVO_UV3F8')

    '''
    import pickle
    model = pickle.load(open('model/saved_model/Lgbm_model.sav', 'rb'))
    #['전설의 도우너', '사스가수수', '아마테라스탈론', 'SaintVansan', '워터스킨']
    #['나 현우 아니다', 'The White Artist', '아마테라스탈론', 'SaintVansan', '황제고양이']
    #value = predict_feature(['SaintVansan']*5, model)
    value = predict_feature(['나 현우 아니다', 'The White Artist', '아마테라스탈론', 'SaintVansan', '황제고양이'], model)
    mean = 0.5
    std = 0.073
    diff = abs(value - mean)
    if diff < std:
        print("Normal")
    elif diff < std * 2:
        print("Abnormal")
    print(value)
    '''