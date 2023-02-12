import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.backbones import algos
import numpy as np

class GraphLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        # self.W = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]
        # return torch.matmul(x, self.W[None, :]) + self.b[None, None, :]

class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, adjmat, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjmat = adjmat
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                output.append(spmm(self.adjmat, support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphResBlock(nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, A):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, A)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))

    def forward(self, x):
        y = F.relu(self.pre_norm(x))
        y = self.lin1(y)

        y = F.relu(self.norm1(y))
        y = self.conv(y.transpose(1,2)).transpose(1,2)

        y = F.relu(self.norm2(y))
        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return x+y



class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """
    def __init__(self, num_joint, num_degree, embed_dim):
        super(GraphNodeFeature, self).__init__()

        self.num_joint = num_joint
        self.embed_dim = embed_dim

        self.atom_encoder = nn.Linear(num_joint * 2, num_joint * embed_dim)
        self.degree_encoder = nn.Embedding(num_degree, embed_dim, padding_idx=0)

    def forward(self, x, adj):
        B = x.shape[0]
        node_feature = self.atom_encoder(x) 
        node_feature = node_feature.reshape(-1, self.num_joint, self.embed_dim)

        x_degree = adj.long().sum(dim=1).view(-1)

        degree_feature = self.degree_encoder(x_degree)
        node_feature = (node_feature + degree_feature)

        return node_feature

class GraphAttnBias_edge(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(self, num_heads=8, num_spatial=10, num_joint=17, spatial_pos=None, edg_adj=None):
        super(GraphAttnBias_edge, self).__init__()
        self.num_heads = num_heads
        self.num_joint = num_joint
        edg_adj[edg_adj == -1] = 0
        self.edg_adj = edg_adj
        num_edges = num_joint + 6

        self.spatial_pos = spatial_pos.long()

        ones = torch.ones_like(spatial_pos)
        self.spatial = spatial_pos - ones
        self.spatial = torch.where(self.spatial > 0, self.spatial, ones)
        self.spatial = self.spatial.expand(num_heads, -1, -1)
        self.spatial = 1.0 / self.spatial

        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.edge_encoder = nn.Linear(num_joint * num_joint, num_joint * num_joint * num_heads)
        self.W = nn.Parameter(torch.ones(num_heads, edg_adj.shape[0], edg_adj.shape[1], edg_adj.shape[2]))

    def reset_parameters(self):
        w_stdv = 1 / (self.num_joint) 
        self.enc.data.uniform_(-w_stdv, w_stdv)

    def forward(self):
        # [num_joint, num_joint, num_head] -> [num_head, num_joint, num_joint]
        spatial_pos_bias = self.spatial_pos_encoder(self.spatial_pos).permute(2, 0, 1)
        edg_adj = self.edg_adj.permute(2, 0, 1) # [max_dist, joint_nums, joint_nums]
        edg_adj = self.edge_encoder(edg_adj.view(-1, self.num_joint * self.num_joint)).reshape(-1, self.num_heads, self.num_joint, self.num_joint)
        # [max_dist, num_heads, num_joint, num_joint] - > [num_heads, num_joint, num_joint, max_dist]
        edg_adj = edg_adj.permute(1, 2, 3, 0)
        edge_feature = torch.mul(self.W, edg_adj)
        edge_bias = edge_feature.sum(-1) # [num_head, num_joint, num_joint]
        edge_bias = torch.mul(edge_bias, self.spatial)

        return spatial_pos_bias + edge_bias

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.): # 
        super().__init__()
        self.num_heads = num_heads
        print('[num_heads]',num_heads)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x, attn_bias):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [3, B, 8, 17, 32]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)  # [B, 8, 17, 32]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, 8, 17, 17]
        
        if attn_bias != None: # self.joint_att.shape = [17, 17]   self.graph_adj.shape = [17, 17] 
            attn_bias = attn_bias.expand(B, -1, -1, -1)
            attn = attn + attn_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP_SE(nn.Module):
    def __init__(self, in_features, in_channel, hidden_features=None):
        super().__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.fc_down1 = nn.Linear(in_features*in_channel, in_channel)
        self.fc_down2 = nn.Linear(in_channel, 2*in_channel)
        self.fc_down3 = nn.Linear(2*in_channel, in_channel)
        self.sigmoid = nn.Sigmoid()

        self.act = nn.GELU()

    def forward(self, x):
        B = x.shape[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        ####up_stream
        x1 = x
        ### down_stream
        x2 = x.view(B,-1)
        x2 = self.fc_down1(x2).view(B,1,-1)
        x2 = self.act(x2)
        x2 = self.fc_down2(x2)
        x2 = self.act(x2)
        x2 = self.fc_down3(x2)
        x2 = self.sigmoid(x2)
        #### out
        x = ((x1.transpose(1,2))*x2).transpose(1,2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, in_channel, hidden_features=None, dropout=0.1):
        super().__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = self.dropout(x) # according to wenhao

        return x

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj.unsqueeze(0)
        self.kernel_size = self.adj.size(0) # 1
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x) # [B, 512, 1, 17]

        n, kc, t, v = x.size() # [B, 512, 1, 17]
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v) # [B, 1, 512, 1, 17]
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj)) # [B, 512, 1, 17]

        return x.contiguous()

class MGCN(nn.Module):
    """
    Semantic graph convolution layer
    """
    def __init__(self, in_features, out_features, in_channel, adj, bias=True):
        super(MGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if adj == None:
            adj = nn.Parameter(torch.eye(in_channel, in_channel, dtype=torch.float, requires_grad = True))

        self.adj = adj

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj) / 2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)