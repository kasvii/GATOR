import os
import math
import torch
import numpy as np
import torch.nn as nn
import os.path as osp
from core.config import cfg
from timm.models.layers import DropPath
from funcs_utils import load_checkpoint
from models.backbones.modules import GraphLinear, GraphNodeFeature, Attention, MLP, MGCN, HopPathEncoding, gen_edg_input, X_Feat
import graph_utils

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
SMPL_MEAN_vertices = osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')

class GATBlock(nn.Module):
    def __init__(self, dim, in_channel, num_heads, mlp_ratio=4., graph_adj=None, spatial_pos=None, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, in_channel=in_channel, hidden_features=mlp_hidden_dim)
        self.adj = graph_adj.cuda()
        self.gcn = MGCN(dim, dim, in_channel, self.adj)
        self.x_feat = X_Feat(input_dim=dim, output_dim=dim, spatial_pos=spatial_pos)

    def forward(self, input):
        x, hop_path_encoding = input
        res = x
        x = self.norm1(x)
        # SDGA
        x = self.drop_path(self.attn(x, hop_path_encoding) + self.gcn(x))
        x = res + self.x_feat(x)
        res = x
        x = self.norm2(x)
        x = res + self.drop_path(self.mlp(x))
        return (x, hop_path_encoding)

class GAT(nn.Module):
    def __init__(self, num_joint=17, embed_dim=256, depth=4, graph_adj=None, GCN_depth=1, J_regressor=None, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0.4, attn_drop_rate=0.4, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, act_layer=None, pretrained=False):
        super(GAT,self).__init__()
        self.num_joint = num_joint
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_size = 3 * num_joint
        self.pos_id_embed = nn.Embedding(self.num_joint + 1, self.embed_dim, padding_idx=0)

        # preprocess topology structure
        if graph_adj != None: # del symmetric edges
            graph_adj = graph_utils.sparse_python_to_torch(graph_adj[-1]).to_dense()
            graph_adj[1, 4], graph_adj[4, 1] = 0, 0
            graph_adj[2, 5], graph_adj[5, 2] = 0, 0
            graph_adj[3, 6], graph_adj[6, 3] = 0, 0
            graph_adj[11, 14], graph_adj[14, 11] = 0, 0
            graph_adj[12, 15], graph_adj[15, 12] = 0, 0
            graph_adj[13, 16], graph_adj[16, 13] = 0, 0
            self.register_buffer('graph_adj', graph_adj)
        else:
            self.graph_adj = None

        self.GLinear = nn.Sequential(GraphLinear(2, 64),
                                nn.GroupNorm(64 // 16, 64), 
                                nn.GELU(),
                                GraphLinear(64, self.embed_dim))
        self.pos_num_embed = nn.Embedding(self.num_joint, embed_dim, padding_idx=0)
        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).unsqueeze(0)
        self.register_buffer('init_vertices', init_vertices)
        template_joint = torch.matmul(J_regressor[None, :, :], init_vertices).squeeze(0)
        
        # 3dpw dataset
        if num_joint == 19:
            lhip_idx = 11
            rhip_idx = 12
            lshoulder_idx = 5
            rshoulder_idx = 6
            pelvis = (template_joint[lhip_idx, :] + template_joint[rhip_idx, :]) * 0.5
            pelvis = pelvis.reshape((1, -1))
            neck = (template_joint[lshoulder_idx, :] + template_joint[rshoulder_idx, :]) * 0.5
            neck = neck.reshape((1, -1))
            template_joint = torch.cat((template_joint, pelvis, neck), dim=0)
            shortest_path_result = np.load('./data/base_data/shortest_path_3dpw.npy')
            path = np.load('./data/base_data/path_3dpw.npy')
        else:
            shortest_path_result = np.load('./data/base_data/shortest_path_h36m.npy')
            path = np.load('./data/base_data/path_h36m.npy')
        
        # calculate hop & path encoding
        edg_d = torch.zeros(num_joint, num_joint)
        dis = []
        pair = []
        cnt = 0
        for i in range(num_joint):
            for j in range(i+1, num_joint):
                if self.graph_adj[i][j] == 1:
                    cnt += 1
                    edg_dis =  math.sqrt(((template_joint[i] - template_joint[j]) ** 2).sum(0))
                    dis.append(edg_dis)
                    pair.append([i, j])
                    edg_d[i][j] = edg_dis 
        edg_adj = edg_d
        max_dist = np.amax(shortest_path_result)
        edge_input = gen_edg_input(max_dist, path, edg_adj).cuda()
        spatial_pos = torch.from_numpy((shortest_path_result)).cuda()
        self.get_hop_path_encoding = HopPathEncoding(num_heads=num_heads, num_spatial=10, num_joint=self.num_joint, spatial_pos=spatial_pos, edg_adj=edge_input)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            GATBlock(
            dim=self.embed_dim , in_channel=self.num_joint, num_heads=num_heads, mlp_ratio=mlp_ratio, graph_adj=graph_adj, spatial_pos = spatial_pos,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(self.embed_dim)
        self.lifter = nn.Linear(self.embed_dim * self.num_joint, 3 * self.num_joint)

        if pretrained:
            self._load_pretrained_model()

    def _load_pretrained_model(self):
        print("Loading pretrained posenet...")
        checkpoint = load_checkpoint(load_dir=cfg.MODEL.posenet_path, pick_best=True)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def forward(self, pose2d):
        B = pose2d.shape[0]
        pose2d = pose2d.view(-1, self.num_joint, 2)    # [B, 17, 2]
        pose2d = pose2d.permute(0, 2, 1)               # [B, 2, 17]
        node_feature = self.GLinear(pose2d)            # [B, embed_dim, 17]
        x = node_feature
        x = x.permute(0, 2, 1)                         # [B, 17, embed_dim]

        pos_id = torch.arange(1, self.num_joint + 1).cuda()
        x = x + self.pos_id_embed(pos_id)
        pos_num = self.graph_adj.long().cuda().sum(dim=1).view(-1)
        x = x + self.pos_num_embed(pos_num)
        
        hop_path_encoding = self.get_hop_path_encoding()
        x, _ = self.blocks((x, hop_path_encoding))
        x = self.norm(x)
        x = self.gelu(x)
        x_out = self.lifter(x.view(B, -1))

        return x_out, x

def get_model(num_joint=17, embed_dim=256, depth=4, graph_adj=None, GCN_depth=1, J_regressor=None, pretrained=False): 
    model = GAT(num_joint, embed_dim, depth, graph_adj, GCN_depth, J_regressor, pretrained=pretrained)
    return model