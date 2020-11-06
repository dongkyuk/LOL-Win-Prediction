from scipy.stats import skew, tstd, tmean


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
    '''
    match_df = pd.read_csv("data/match_gold.csv")
    match_df = feature_extraction(match_df)
    match_df.to_csv('data/match_feature_gold.csv', index=False)
    '''