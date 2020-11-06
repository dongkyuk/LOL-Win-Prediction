from league_data import LeagueData
from feature_extraction import feature_extraction
import logging

def main():
    # Set logging level
    logging.basicConfig(level = logging.INFO)

    # Init League data and process info 
    silver1_1 = LeagueData(api_key, 'kr', 'RANKED_SOLO_5x5', 'SILVER', 'I', 1)
    silver1_1.process_info()
    silver1_1.save_info()

    # Feature Extraction
    feature_extraction(silver1_1.match_playerid_df)

    # Train/Test Model
    

def test():
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

