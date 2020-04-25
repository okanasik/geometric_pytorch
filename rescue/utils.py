# import torch
# from torch_scatter import scatter_max, scatter_sum, scatter_min
#
# values = torch.tensor([1, 0, 1, 0, 1, 2, 1, 2, 0, 1])
# print(values)
# indexes = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
# print(indexes)
#
# from torch_scatter import scatter_max, scatter_sum, scatter_min
# _, scatter_indexes = scatter_min(indexes, indexes)
# print(scatter_indexes)
#
# max_values, max_indexes = scatter_max(values, indexes)
# print(max_values)
# print(max_indexes)
# print(max_indexes-scatter_indexes)
# sum_indexes = scatter_sum(indexes, indexes)
# print(sum_indexes)
# all_idx_count = 0
# for i, idx in enumerate(sum_indexes):
#     if i == 0: continue
#     sum_indexes[i] = idx / i
#     all_idx_count += sum_indexes[i]
#
# zero_count = len(indexes) - all_idx_count
# sum_indexes[0] = zero_count
# print(sum_indexes)
from datetime import datetime


def convert_to_datetime(datetime_str):
    if not datetime_str: return None
    datetime_values = [2020, 1, 1, 0, 0, 0] # [year, month, day, hour, min, sec]
    datetime_values[0] = int(datetime_str[0:4])
    datetime_str = datetime_str[4:]
    idx = 1
    while datetime_str and idx < len(datetime_values):
        datetime_values[idx] = int(datetime_str[0:2])
        datetime_str = datetime_str[2:]
        idx += 1

    return datetime(year=datetime_values[0], month=datetime_values[1], day=datetime_values[2], hour=datetime_values[3],
                    minute=datetime_values[4], second=datetime_values[5])


file_name = 'robocup2019_Kobe1_ait_ambulanceteam_280282693_20200422140450.zip'
print(file_name[-3:])