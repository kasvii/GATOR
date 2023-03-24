import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import MDR, GAT


class GATOR(nn.Module):
    def __init__(self, num_joint, embed_dim, depth, graph_adj, GCN_depth, J_regressor):
        super(GATOR, self).__init__()

        self.num_joint = num_joint
        self.pose_lifter = GAT.get_model(num_joint, embed_dim, depth, graph_adj, GCN_depth, J_regressor, pretrained=cfg.MODEL.posenet_pretrained)
        self.pose2mesh = MDR.get_model(num_joint, embed_dim)

    def forward(self, pose2d):
        pose3d, pose3d_feat = self.pose_lifter(pose2d.view(len(pose2d), -1))
        pose3d = pose3d.reshape(-1, self.num_joint, 3)
        pose_combine = torch.cat((pose2d, pose3d / 1000, pose3d_feat), dim=2)
        cam_mesh = self.pose2mesh(pose_combine)

        return cam_mesh, pose3d

def get_model(num_joint, embed_dim, depth, graph_adj, GCN_depth, J_regressor):
    model = GATOR(num_joint, embed_dim, depth, graph_adj, GCN_depth, J_regressor)

    return model


