import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_all_edges(path, i, j):
    k = path[i][j]
    if k == 510:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edg_input(max_dist, path, edge_feat):
    (nrows, ncols) = path.shape
    assert nrows == ncols
    n = nrows
    edge_fea_all = torch.zeros([n, n, max_dist])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path[i][j] == 510:
                continue
            path_ij = [i] + get_all_edges(path, i, j) + [j]
            num_path = len(path_ij) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k] = edge_feat[path_ij[k], path_ij[k+1]]
    
    return edge_fea_all

class GraphLinear(nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs 
    """
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

class HopPathEncoding(nn.Module):
    """
    Compute hop and path encodings for each head.
    """
    def __init__(self, num_heads=8, num_spatial=10, num_joint=17, spatial_pos=None, edg_adj=None):
        super(HopPathEncoding, self).__init__()
        self.num_heads = num_heads
        self.num_joint = num_joint
        edg_adj[edg_adj == -1] = 0
        self.edg_adj = edg_adj

        self.spatial_pos = spatial_pos.long()
        ones = torch.ones_like(spatial_pos)
        self.spatial = spatial_pos - ones
        self.spatial = torch.where(self.spatial > 0, self.spatial, ones)
        self.spatial = self.spatial.expand(num_heads, -1, -1)
        self.spatial = 1.0 / self.spatial
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.edge_encoder = nn.Linear(num_joint * num_joint, num_joint * num_joint * num_heads)
        self.W = nn.Parameter(torch.ones(num_heads, edg_adj.shape[0], edg_adj.shape[1], edg_adj.shape[2]))

    def forward(self):
        spatial_pos_bias = self.spatial_pos_encoder(self.spatial_pos).permute(2, 0, 1)
        edg_adj = self.edg_adj.permute(2, 0, 1)
        edg_adj = self.edge_encoder(edg_adj.view(-1, self.num_joint * self.num_joint)).reshape(-1, self.num_heads, self.num_joint, self.num_joint)
        edg_adj = edg_adj.permute(1, 2, 3, 0)
        edge_feature = torch.mul(self.W, edg_adj)
        edge_bias = edge_feature.sum(-1)
        edge_bias = torch.mul(edge_bias, self.spatial)

        return spatial_pos_bias + edge_bias

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.): # 
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, attn_bias):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_bias != None: 
            attn_bias = attn_bias.expand(B, -1, -1, -1)
            attn = attn + attn_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class X_Feat(nn.Module):
    def __init__(self, input_dim, output_dim, s=1, l=2, d=8, spatial_pos=None): 
        super().__init__()
        self.output_patch = 1 + l - s
        self.s = s
        self.l = l
        self.spatial_pos = spatial_pos
        self.linears = nn.ModuleList()
        c_out = int(input_dim)
        c_out_total = int(0)
        for k in range(s, l+1, 1):
            c_out_total = c_out_total + c_out
            self.linears.append(
                nn.Linear(input_dim, c_out)
            )
            c_out = int(c_out / d)
        self.linearback = nn.Linear(int(c_out_total), int(output_dim))

    def forward(self, input):
        B = input.shape[0]
        H_features = []
        ones = torch.ones_like(self.spatial_pos).cuda()
        zeros = torch.zeros_like(self.spatial_pos).cuda()
        for k in range(self.s, self.l+1, 1):
            new_channel = self.linears[k - self.s](input)
            if k == self.s:
                mask = torch.where(self.spatial_pos <= k, ones, zeros)
            else:
                mask = torch.where(self.spatial_pos == k, ones, zeros)
            mask = mask.expand(B, -1, -1).cuda()
            mask = mask.type_as(new_channel)
            new_feature_sum = torch.bmm(mask, new_channel)
            H_features.append(new_feature_sum)
        
        features = torch.cat(H_features, dim=-1)
        features = self.linearback(features)         
        
        return features

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
        x = self.dropout(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj.unsqueeze(0)
        self.kernel_size = self.adj.size(0) # 1
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)                                            # [B, embed_dim, 1, 17]
        n, kc, t, v = x.size()                                      # [B, embed_dim, 1, 17]
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v) # [B, 1, embed_dim, 1, 17]
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj))         # [B, embed_dim, 1, 17]

        return x.contiguous()

class MGCN(nn.Module):
    """
    Modulated graph convolution from https://github.com/ZhimingZo/Modulated-GCN/blob/main/Modulated_GCN/Modulated_GCN_gt/models/modulated_gcn_conv.py
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