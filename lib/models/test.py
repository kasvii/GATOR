import os
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

num_joint = 3
edg_dis = torch.rand(num_joint, num_joint)
edg_adj = torch.zeros([num_joint, num_joint, 1], dtype=torch.long)

dis = []
pair = []
for i in range(num_joint):
    for j in range(num_joint):
        dis.append(edg_dis[i][j])
        pair.append([i, j])

print('[dis] ', dis)
print('[pair] ', pair)

dis = torch.Tensor(dis)
pair = torch.tensor(pair)
edg_idx = torch.argsort(dis)
print('[edg_idx] ', edg_idx)
# pair = torch.cat(pair, 0)
print('[pair] ', pair)

edg_adj[pair[edg_idx,0], pair[edg_idx,1]] = torch.arange(num_joint * num_joint, 0,  -1, dtype=torch.long).unsqueeze(1)
print('[pair[:, 0]] ', pair[:, 0])
print('[pair[:, 1]] ', pair[:, 1])
print('[edg_adj] ', edg_adj)