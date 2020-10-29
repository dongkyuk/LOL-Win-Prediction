from itertools import permutations
from scipy.stats import skew, tstd, tmean

'''
example_str = "TORADORA님이 방에 참가했습니다. NuggeTnT님이 방에 참가했습니다. 소꼬기님이 방에 참가했습니다. 두두갓님이 방에 참가했습니다. 동글님이 방에 참가했습니다"

def parse_multi_search(input_str):
    str_lst = input_str.split()
    suffix = "님이"
    summoner_set = set()
    for str in str_lst:
        if str.endswith(suffix):
             summoner_set.add(str[:-2])
    return list(summoner_set)

summoner_lst = parse_multi_search(example_str)
print(summoner_lst)

specify_lane = False
if not specify_lane:
    permute = permutations(summoner_lst,5)
print(list(permute))
'''

data = [1] * 100 + [0] * 50
win_skew = skew(data)
win_std = tstd(data)
print(win_skew)
win_mean = tmean(data)