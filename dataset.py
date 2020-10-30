from riotwatcher import LolWatcher, ApiError, TftWatcher
from utils.utils import print_if_complete
import pandas as pd
import json
import collections
from scipy.stats import skew, tstd, tmean
from time import sleep
from progress.bar import Bar

# Global variables
api_key = 'RGAPI-1a444448-c73e-4fa4-9b7b-44c31ce70f98'
watcher = LolWatcher(api_key)
my_region = 'kr'
my_summoner_id = 'HardcoreZealot'


@print_if_complete
def league_id_lst(league):
    # Get user puuid list in certain league
    id_lst = list()
    with Bar('Processing...') as bar:
        for summoner_info in league:
            id = summoner_info['summonerId']
            summoner = watcher.summoner.by_id(my_region, id)
            id_lst.append(summoner['accountId'])
            bar.next()
    return id_lst


@print_if_complete
def unique_match_id_lst(id_lst, count=10):
    # Get unqiue match id lst from a given list of user puuid
    res_lst = list()
    with Bar('Processing...') as bar:
        for id in id_lst:
            try:
                match_lst = watcher.match.matchlist_by_account(
                    my_region, id, '420', end_index=count)
            except:
                continue
            for match in match_lst['matches']:
                res_lst.append(match['gameId'])
            res_lst = list(set(res_lst))  # Remove Duplicates
            bar.next()
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

    with Bar('Processing...', max=len(match_lst)-1) as bar:
        for index, match_data in enumerate(match_lst[4030:]):
            for role_index in range(2, 7):
                match_data = match_data + \
                    list(player_info(match_data[role_index]))
            match_lst[index+1] = match_data
            bar.next()
            match_df = pd.DataFrame(
                match_lst[1:index+1], columns=match_lst[0]+new_col)
            match_df = match_df.drop(["match_id", "bot_support",
                                      "bot_carry", "mid", "jungle", "top"], axis=1)
            match_df.to_csv('data/match_feature_2.csv', index=False)

    return match_df


def main():
    '''
    silver1_1 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 1)
    silver1_2 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 2)
    silver1_3 = watcher.league.entries(
        my_region, 'RANKED_SOLO_5x5', 'SILVER', 'I', 3)

    silver1 = silver1_1 + silver1_2 + silver1_3

    id_lst = league_id_lst(silver1)
    with open("data/id_lst.txt", "w") as fp:
        json.dump(id_lst, fp)

    with open("data/id_lst.txt", "r") as fp:
        id_lst = json.load(fp)

    match_id_lst = unique_match_id_lst(id_lst)
    with open("data/match_id_lst.txt", "w") as fp:
        json.dump(match_id_lst, fp)

    with open("data/match_id_lst.txt", "r") as fp:
        match_id_lst = json.load(fp)
    match_df = match_info(match_id_lst)
    match_df.to_csv('data/match.csv', index=False)
    '''
    match_df = pd.read_csv("data/match.csv")
    match_df = feature_extraction(match_df)
    match_df.to_csv('data/match_feature.csv', index=False)


if __name__ == "__main__":
    main()
    # me = watcher.summoner.by_id(my_region, my_summoner_id)
    # accountId = me['accountId']
    # player_info('pRzYZSsbfU4Ha0SlczOED7iT_mtDn-BuKvaLMZNNP1w8ZfEvVO_UV3F8')
