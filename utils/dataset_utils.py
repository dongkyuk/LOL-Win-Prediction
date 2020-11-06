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