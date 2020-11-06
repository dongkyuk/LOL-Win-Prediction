import pandas as pd
import json
from multiprocessing.dummy import Pool as ThreadPool
from riotwatcher import LolWatcher, ApiError, TftWatcher
from utils.decorator import log_process
from utils.dataset_utils import append_match_info_lst


class LeagueData():
    """
        A class used to represent datas of a certain league

        Attributes
        ----------
        account_id_lst - list of account ids in the league
        match_id_lst - list of match ids in the league
        match_playerid_df - dataframe of matches and corresponding player account ids

        Methods
        -------
        process_info : process the infos
        save_info : save the processed infos
    """

    def __init__(self, api_key, region, game_type, tier, division, page):
        """
            Parameters
            ----------
            api_key - Riot api key
            region (string) – the region to execute this request on
            queue (string) – the queue to query, i.e. RANKED_SOLO_5x5
            tier (string) – the tier to query, i.e. DIAMOND
            division (string) – the division to query, i.e. III
            page (int) – the page for the query to paginate to. Starts at 1
        """
        self.watcher = LolWatcher(api_key)
        self.region = region
        self.league_entry_lst = self.watcher.league.entries(
            region, game_type, tier, division, page)

        self.account_id_path = "data/account_id/" + region + \
            game_type + tier + division + page + ".txt"
        self.match_id_path = "data/match_id/" + region + \
            game_type + tier + division + page + ".txt"
        self.match_playerid_path = "data/match_playerid/" + \
            region + game_type + tier + division + page + ".csv"

    @log_process
    def process_info(self, read_existing=True):
        # Process all information, if read_existing, read the existing files instead
        if read_existing:
            with open(self.account_id_path, "r") as fp:
                self.account_id_lst = json.load(fp)
            with open(self.match_id_path, "r") as fp:
                self.match_id_lst = json.load(fp)
            self.match_playerid_df = pd.read_csv(self.match_playerid_path)
        else:
            self.account_id_lst = self._get_account_id_lst()
            self.match_id_lst = self._get_recent_match_id_lst()
            self.match_playerid_df = self._get_match_playerid_df()

    @log_process
    def save_info(self):
        with open(self.account_id_path, "w") as fp:
            json.dump(self.account_id_lst, fp)
        with open(self.match_id_path, "w") as fp:
            json.dump(self.match_id_lst, fp)
        self.match_playerid_df.to_csv(self.match_playerid_path, index=False)

    @log_process
    def _get_account_id_lst(self, threading=False):
        """ 
            Inputs: 
                League Entry List
                threading config (bool)
            Outputs: 
                AccountId List 
        """

        def _get_league_id(league_entry):
            # Get single league id from league entry
            summoner_id = league_entry['summonerId']

            try:
                summoner = self.watcher.summoner.by_id(
                    self.region, summoner_id)
            except:
                return None

            return summoner['accountId']

        if threading:
            pool = ThreadPool(3)
            account_id_lst = pool.map(_get_league_id, self.league_entry_lst)
        else:
            account_id_lst = list(map(_get_league_id, self.league_entry_lst))

        return account_id_lst

    @log_process
    def _get_recent_match_id_lst(self, recent_n=10):
        """ 
            Inputs: 
                AccountId List
                recent_n (int)
            Outputs: 
                Unique match id list of recent_n games of given accounts
        """

        # Init empty match id list
        match_id_lst = list()

        for account_id in self.account_id_lst:
            try:
                # 420 makes sure we only collect ranked solo q
                match_lst = self.watcher.match.matchlist_by_account(
                    self.region, account_id, '420', end_index=recent_n)
            except:
                continue

            # Update match id list
            match_id_lst = match_id_lst + \
                [match['gameId'] for match in match_lst]
            # Remove Duplicates
            match_id_lst = list(set(match_id_lst))

        return match_id_lst

    @log_process
    def _get_match_playerid_df(self):
        """ 
            Inputs: 
                matchId List
            Outputs: 
                dataframe of match and corresponding playerid with following columns
                ["match_id", "team", "bot_support_id",
                "bot_carry_id", "mid_id", "jungle_id", "top_id", "win"] 
        """

        # Init match playerid list with columns of dataframe
        match_playerid_lst = [["match_id", "team", "bot_support",
                               "bot_carry", "mid", "jungle", "top", "win"]]

        for match_id in self.match_id_lst:
            try:
                match = self.watcher.match.by_id(self.region, match_id)
            except:
                continue
            # Append match playerid list for red team
            match_playerid_lst = append_match_info_lst(
                100, match_playerid_lst, match_id, match)
            # Append match playerid list for blue team
            match_playerid_lst = append_match_info_lst(
                200, match_playerid_lst, match_id, match)

        # Make dataframe from list
        match_playerid_df = pd.DataFrame(
            match_playerid_lst[1:], columns=match_playerid_lst[0])

        return match_playerid_df
