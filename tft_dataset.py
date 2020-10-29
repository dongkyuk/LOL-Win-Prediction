from riotwatcher import LolWatcher, ApiError, TftWatcher
from utils.utils import print_if_complete
import pandas as pd
import json
import collections
from flatten_dict import flatten

# Global variables
api_key = 'RGAPI-767133c2-cd2b-434d-8c0f-0ce87e4cda40'
watcher = TftWatcher(api_key)
my_region = 'kr'
file = open("MyFile.txt", "a")


@print_if_complete
def league_puuid_lst(league):
    # Get user puuid list in certain league
    puuid_lst = list()
    for summoner_info in league['entries']:
        summonerId = summoner_info['summonerId']
        puuid = watcher.summoner.by_id(my_region, summonerId)['puuid']
        puuid_lst.append(puuid)
    return puuid_lst


@print_if_complete
def unique_match_id_lst(puuid_lst, count=10):
    # Get unqiue match id lst from a given list of user puuid
    res_lst = list()
    for puuid in puuid_lst:
        match_id_lst = watcher.match.by_puuid('asia', puuid, count=10)
        res_lst = res_lst + match_id_lst
        res_lst = list(set(res_lst))  # Remove Duplicates
    return res_lst


@print_if_complete
def match_info(match_id_lst):
    # Return Dataframe of match info from a given list of match ids
    for match_id in match_id_lst:
        match = watcher.match.by_id('asia', match_id)
        temp_df = pd.DataFrame(data=match)
        match_df = match_df.append(temp_df)
    return match_df


def main():
    # Get league info
    challenger = watcher.league.challenger(my_region)
    puuid_lst = league_puuid_lst(challenger)
    json.dump(puuid_lst, file)
    match_id_lst = unique_match_id_lst(puuid_lst)
    json.dump(match_id_lst, file)
    #match_df = match_info(match_id_lst)

    # file1.write(json.dumps(challenger['entries']))


if __name__ == "__main__":
    # main()
    match = watcher.match.by_id('asia', "KR_4745071686")
    print(match)
    match = flatten(match, reducer='underscore')

    # print(match['metadata_participants'])
    #match = match['info_participants']['traits']
    #traits = info_participants['traits']
    #units = info_participants['units']
    #match['info_participants'] = list(map(flatten, match['info_participants']))

    temp_df = pd.DataFrame(match)
    temp_df.to_csv('match.csv', index=False)


    
