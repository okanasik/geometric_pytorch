# mypath = "/home/okan/robotics/roborescue/rcrs-server/boot/config/comms"
# mypath = "/home/okan/robotics/roborescue/rcrs-server/boot/traffic3/data/"

# import torch
# from torch_geometric.datasets import TUDataset
# import os.path as osp
#
# class HandleNodeAttention(object):
#     def __call__(self, data):
#         data.attn = torch.softmax(data.x[:, 0], dim=0)
#         data.x = data.x[:, 1:]
#         return data
#
#
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'COLORS-3')
# dataset = TUDataset(path, 'COLORS-3', use_node_attr=True,
#                     transform=HandleNodeAttention())
