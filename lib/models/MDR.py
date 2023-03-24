import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from core.config import cfg
import graph_utils
from graph_utils import build_verts_joints_relation

from models.backbones.mesh import Mesh
from models.vanilla_transformer_encoder import MultiHeadedAttention, LayerNorm

import os.path as osp

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR

class CrossAttention(nn.Module):
    def __init__(self, dim, joint_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.joint_num = joint_num
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        vert_num = N - self.joint_num
        q = self.wq(x[:, 0:vert_num, ...]).reshape(B, vert_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x[:, - self.joint_num:, ...]).reshape(B, self.joint_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x[:, - self.joint_num:, ...]).reshape(B, self.joint_num, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, vert_num, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, joint_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2, attn_drop=0.2,
                 drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.joint_num = joint_num
        self.attn = CrossAttention(
            dim, joint_num = joint_num, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        vert_num = x.shape[1] - self.joint_num
        x = x[:, 0:vert_num, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MDR(nn.Module):
    def __init__(self, num_joint, embed_dim, SMPL_MEAN_vertices=osp.join(BASE_DATA_DIR, 'smpl_mean_vertices.npy')):
        super(MDR, self).__init__()
        self.embed_dim = 64
        self.num_joint = num_joint
        dropout=0.1
        self.mesh = Mesh()
        
        init_vertices = torch.from_numpy(np.load(SMPL_MEAN_vertices)).cuda()
        downsample_verts_1723 = self.mesh.downsample(init_vertices) # [1723, 3]
        downsample_verts_431 = self.mesh.downsample(downsample_verts_1723, n1=1, n2=2) # [431, 3]
        self.register_buffer('init_vertices', downsample_verts_431)
        self.register_buffer('init_vertices_6890', init_vertices)
        
        J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy')).astype(np.float32)).cuda()
        self.joints_template = torch.matmul(J_regressor, init_vertices)
        self.vj_relation, self.jv_sets = build_verts_joints_relation(self.joints_template.cpu().numpy(), downsample_verts_431.cpu().numpy())

        self.joints_template = self.joints_template.cuda()
        self.num_verts = downsample_verts_431.shape[0]

        self.pos_j_id_embed = nn.Embedding(self.num_joint + 1, self.embed_dim, padding_idx=0)
        self.pos_v_id_embed = nn.Embedding(self.num_verts + 1, self.embed_dim, padding_idx=0)

        # LBF module
        self.encoder = CrossAttentionBlock(dim = self.embed_dim, joint_num = self.num_joint, num_heads=2)
        self.selfatt = MultiHeadedAttention(h = 2, d_model = self.embed_dim, dropout=0.1)
        self.norm = LayerNorm(self.embed_dim)

        self.encoder_1 = CrossAttentionBlock(dim = self.embed_dim, joint_num = self.num_joint, num_heads=2)
        self.selfatt_1 = MultiHeadedAttention(h = 2, d_model = self.embed_dim, dropout=0.1)
        self.norm_1 = LayerNorm(self.embed_dim)

        self.encoder_2 = CrossAttentionBlock(dim = self.embed_dim, joint_num = self.num_joint, num_heads=2)
        self.selfatt_2 = MultiHeadedAttention(h = 2, d_model = self.embed_dim, dropout=0.1)
        self.norm_2 = LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.get_joint_feature = nn.Linear(2 + 3 + embed_dim, self.embed_dim)
        self.get_verts_feature = nn.Linear(3 + 3, self.embed_dim)
        
        # MDR Head
        self.motion_linear = nn.Linear(self.embed_dim, 23)
        self.bias_linear = nn.Linear(self.embed_dim, 3)
        if cfg.MODEL.alpha:
            self.bias_norm = nn.LayerNorm(3)
            self.scale_linear = nn.Linear(self.embed_dim, 1)
        else:
            self.bias_norm = nn.BatchNorm1d(self.num_verts)
        self.bias_gelu = nn.GELU()
        self.bias_conv1d = nn.Conv1d(self.num_verts, 20, kernel_size=3, padding=1)
        self.upsample_conv = nn.Conv1d(self.num_verts, 6890, kernel_size=3, padding=1)

    def forward(self, x):
        B = x.shape[0]
        joint_feat = x
        verts_feat = self.init_vertices.unsqueeze(0).expand(B, -1, -1) 
        verts_feat = torch.cat([verts_feat, x[:, self.vj_relation, 2:5]], dim = 2) # [B, 431, 3 + 3 + 2 + 3 + embed_dim]

        joint_feat = self.get_joint_feature(joint_feat)
        verts_feat = self.get_verts_feature(verts_feat)

        pos_id_joint = torch.arange(1, self.num_joint + 1).cuda()
        joint_feat = joint_feat + self.pos_j_id_embed(pos_id_joint) # [B, 17,  embed_dim]

        pos_id_verts = torch.arange(1, self.num_verts + 1).cuda()
        verts_feat = verts_feat + self.pos_v_id_embed(pos_id_verts) # [B, 431, embed_dim]
        
        # LBF module
        fusion_feat = torch.cat([verts_feat, joint_feat], dim = 1)
        verts_feat = self.encoder(fusion_feat)
        verts_feat = self.norm(verts_feat) 
        verts_feat = verts_feat + self.dropout(self.selfatt(verts_feat, verts_feat, verts_feat))

        fusion_feat = torch.cat([verts_feat, joint_feat], dim = 1)
        verts_feat = self.encoder_1(fusion_feat)
        verts_feat = self.norm_1(verts_feat) 
        verts_feat = verts_feat + self.dropout(self.selfatt_1(verts_feat, verts_feat, verts_feat))

        fusion_feat = torch.cat([verts_feat, joint_feat], dim = 1)
        verts_feat = self.encoder_2(fusion_feat)
        verts_feat = self.norm_2(verts_feat) 
        verts_feat = verts_feat + self.dropout(self.selfatt_2(verts_feat, verts_feat, verts_feat))

        # MDR Head  [B, 431, embed_dim] -> [B, 6890, 3]
        AC_feat = self.motion_linear(verts_feat)                      # [B, 431, embed_dim] -> [B, 431, 23]
        mat_A, mat_C = AC_feat[:, :, :20], AC_feat[:, :, -3:]         # [B, 431, 20] [B, 431, 3]
        mat_B = self.bias_linear(verts_feat)                          # [B, 431, embed_dim] -> [B, 431, 3]
        mat_B = self.bias_norm(mat_B)
        mat_B = self.bias_gelu(mat_B)
        mat_B = self.bias_conv1d(mat_B)                               # [B, 431, 3] -> [B, 20, 3]
        if cfg.MODEL.alpha:
            alpha = 1.1 ** self.scale_linear(verts_feat)              # [B, 431, embed_dim] -> [B, 431, 1]
        else:
            alpha = 1
        vert_coor = alpha * mat_A.softmax(dim=-1).bmm(mat_B) + mat_C  # [B, 431, 3]
        vert_coor = self.upsample_conv(vert_coor)                     # [B, 6890, 3]
        vert_coor = vert_coor + self.init_vertices_6890

        return vert_coor                                              # [B, 6890, 3]

def get_model(num_joint, embed_dim):
    model = MDR(num_joint, embed_dim)
    return model

def test_net():
    batch_size = 3
    model = get_model(17, 128)
    model = model.cuda()
    model.eval()
    input = torch.randn(batch_size, 17, 2 + 3 + 128).cuda()
    pred = model(input)
    print(pred.shape)

if __name__ == '__main__':
    test_net()
