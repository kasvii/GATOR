import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import colorsys
import json
import argparse

import __init_path
import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from graph_utils import build_coarse_graphs
from renderer import Renderer
from vis import vis_2d_keypoints, vis_coco_skeleton
from _mano import MANO
from smpl import SMPL

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w/2, y + h/2, h
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def render(result, orig_height, orig_width, orig_img, mesh_face, color):
    pred_verts, pred_cam, bbox = result['mesh'], result['cam_param'][None, :], result['bbox'][None, :]

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bbox,
        img_width=orig_width,
        img_height=orig_height
    )

    # Setup renderer for visualization
    renderer = Renderer(mesh_face, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    renederd_img = renderer.render(
        orig_img,
        pred_verts,
        cam=orig_cam[0],
        color=color,
        mesh_filename=None,
        rotate=False
    )

    return renederd_img


def get_joint_setting(mesh_model, joint_category='human36'):
    joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = None, None, None, None, None
    if joint_category == 'human36':
        joint_regressor = mesh_model.joint_regressor_h36m
        joint_num = 17
        skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        graph_Adj, graph_L, graph_perm,graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)

        model_chk_path = './results/h36m_det.pth.tar'

    elif joint_category == 'coco':
        joint_regressor = mesh_model.joint_regressor_coco
        joint_num = 19  # add pelvis and neck
        skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './results/3dpw_det.pth.tar'
        
    else:
        raise NotImplementedError(f"{joint_category}: unknown joint set category")

    # model = models.pose2mesh_net.get_model(joint_num, graph_L)
    joint_regressor = torch.Tensor(joint_regressor)
    model = models.Graphormer.get_model(joint_num, embed_dim=128, depth=6, graph_adj=graph_Adj, GCN_depth=2, J_regressor=joint_regressor)
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse


def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for pelvis
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))
    return joint_coord

def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]  # confidence for neck
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))
    return joint_coord


def optimize_cam_param(project_net, joint_input, crop_size):
    bbox = get_bbox(joint_input)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(joint_input.copy(), (crop_size, crop_size), bbox1, 0, 0, None)
    joint_img, _ = j2d_processing(joint_input.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]), bbox2, 0, 0, None)

    joint_img = joint_img[:, :2]
    joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = torch.Tensor(joint_img[None, :, :]).cuda()
    target_joint = torch.Tensor(proj_target_joint_img[None, :, :2]).cuda()

    # get optimization settings for projection
    criterion = nn.L1Loss()
    optimizer = optim.Adam(project_net.parameters(), lr=0.1)

    # estimate mesh, pose
    model.eval()
    pred_mesh = model(joint_img)
    pred_3d_joint = torch.matmul(joint_regressor, pred_mesh)

    out = {}
    # assume batch=1
    project_net.train()
    for j in range(0, 1500):
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())

        loss = criterion(pred_2d_joint, target_joint[:, :17, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j == 500:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.05
        if j == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    out['cam_param'] = project_net.cam_param[0].detach().cpu().numpy()
    out['bbox'] = bbox1

    out['target'] = proj_target_joint_img

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render Pose2Mesh output')
    parser.add_argument('--gpu', type=str, default='0', help='assign gpu number')
    parser.add_argument('--input_pose', type=str, default='.', help='path of input 2D pose')
    parser.add_argument('--input_img', type=str, default='.', help='path of input image')
    parser.add_argument('--joint_set', type=str, default='coco', help='choose the topology of input 2D pose from [human36, coco, smpl, mano]')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    virtual_crop_size = 500
    joint_set = args.joint_set
    input_path = args.input_pose
    output_path = './demo/result/'
    cfg.DATASET.target_joint_set = joint_set
    cfg.MODEL.posenet_pretrained = False

    # prepare model
    if joint_set == 'mano':
        mesh_model = MANO()
    else:
        mesh_model = SMPL()
    model, joint_regressor, joint_num, skeleton, graph_L, graph_perm_reverse = get_joint_setting(mesh_model, joint_category=joint_set)
    model = model.cuda()
    joint_regressor = torch.Tensor(joint_regressor).cuda()

    if input_path != '.':  # user specific input
        coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        # get camera parameters
        project_net = models.project_net.get_model(crop_size=virtual_crop_size).cuda()
        joint_input = np.array([[1291.3638916015625, 133.28578186035156, 0.9982743263244629], [1314.344482421875, 115.85456848144531, 0.9863348007202148], [1280.970458984375, 115.5665283203125, 0.9855356216430664], [1362.3145751953125, 128.57400512695312, 0.9809767007827759], [1271.8909912109375, 122.30479431152344, 0.8529441952705383], [1416.35693359375, 237.69175720214844, 0.941078782081604], [1223.92333984375, 227.0111083984375, 0.9616540670394897], [1463.168212890625, 375.3280334472656, 0.9519579410552979], [1162.106201171875, 343.2330322265625, 0.9806461930274963], [1427.9552001953125, 461.90924072265625, 0.9899344444274902], [1146.637451171875, 450.2824401855469, 0.9859115481376648], [1350.5384521484375, 523.412353515625, 0.8887602686882019], [1244.294677734375, 517.5330200195312, 0.9031262397766113], [1283.4971923828125, 713.2562255859375, 0.8686809539794922], [1293.2506103515625, 721.4976196289062, 0.9003548622131348], [1206.38525390625, 861.4423828125, 0.919001579284668], [1322.1827392578125, 949.37841796875, 0.917795717716217]])
        joint_input = joint_input.reshape(17,3)
        if joint_set == 'coco':
            joint_input = add_pelvis(joint_input, coco_joints_name)
            joint_input = add_neck(joint_input, coco_joints_name)
            joint_input = joint_input[:,:2]

        if args.input_img != '.':

            orig_img = cv2.imread(args.input_img)
            orig_width, orig_height = orig_img.shape[:2]
        else:
            orig_width, orig_height = int(np.max(joint_input[:, 0]) * 1.5), int(np.max(joint_input[:, 1]) * 1.5)
            orig_img = np.zeros((orig_height, orig_width,3))

        out = optimize_cam_param(project_net, joint_input, crop_size=virtual_crop_size)
        np.save('./demo/output/joint_input', joint_input)
        np.save('./demo/output/out', out)
        np.save('./demo/output/orig_height', orig_height)
        np.save('./demo/output/orig_width', orig_width)
        np.save('./demo/output/orig_img', orig_img)
        np.save('./demo/output/mesh_model', mesh_model.face)
        
        # vis mesh
        # color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        # np.save('./demo/color', color)
        # rendered_img = render(out, orig_height, orig_width, orig_img, mesh_model.face, color)  # s[idx])
        # cv2.imwrite(output_path + f'demo_mesh.png', rendered_img)

        # vis 2d pose
        # tmpkps = np.zeros((3, len(joint_input)))
        # tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = joint_input[:, 0], joint_input[:, 1], 1
        # tmpimg = orig_img.copy().astype(np.uint8)
        # pose_vis_img = vis_2d_keypoints(tmpimg, tmpkps, skeleton)
        # cv2.imwrite(output_path + f'demo_pose2d.png', pose_vis_img)

        # save_obj(out['mesh'], mesh_model.face, output_path + f'demo_mesh.obj')

    else:  # demo on CrowdPose dataset samples, dataset paper link: https://arxiv.org/abs/1812.00324
        # only for vis
        vis_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
        vis_skeleton = ((0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18), (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

        # prepare input image and 2d pose
        coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        pose2d_result_path = './demo/demo_input_2dpose.json'  # coco 2d pose detection detected by HigherHRNet
        with open(pose2d_result_path) as f:
            pose2d_result = json.load(f)
        img_dir = './demo/demo_input_img'
        img_name = '106542.jpg'  # '101570.jpg'
        img_path = osp.join(img_dir, img_name)
        orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        orig_height, orig_width = orig_img.shape[:2]
        pose_vis_img = orig_img.copy()
        coco_joint_list = pose2d_result[img_name]

        # set depth order and colors of people
        c = coco_joint_list
        # coco_joint_list = [c[1], c[0], c[3], c[2]]
        # colors = [(0.9797650775322294, 1.0, 0.5), (0.6383497919730772, 0.5, 1.0), (1.0, 0.6607429415992977, 0.5), (0.5, 0.5997459388041508, 1.0)]

        # manual nms for hhrnet 2d pose input, openpose input may not need this process
        det_score_thr = 1.5
        min_diff = 20

        drawn_joints = []
        for idx in range(len(coco_joint_list)):
            # filtering
            pose_thr = 0.1
            coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]
            coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
            coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
            coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
            # filter inaccurate inputs
            det_score = sum(coco_joint_img[:, 2])
            if det_score < 1.5:
                continue
            # filter filter the same targes
            tmp_joint_img = coco_joint_img.copy()
            continue_check = False
            for ddx in range(len(drawn_joints)):
                drawn_joint_img = drawn_joints[ddx]
                drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
                diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
                diff = diff[diff != 0]
                if diff.size == 0:
                    continue_check = True
                elif diff.mean() < min_diff:
                    continue_check = True
            if continue_check:
                continue
            drawn_joints.append(tmp_joint_img)

            # get camera parameters
            project_net = models.project_net.get_model(crop_size=virtual_crop_size).cuda()
            joint_input = coco_joint_img
            out = optimize_cam_param(project_net, joint_input, crop_size=virtual_crop_size)

            # vis mesh
            # color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
            # orig_img = render(out, orig_height, orig_width, orig_img, mesh_model.face, color)#s[idx])
            # cv2.imwrite(output_path + f'{img_name[:-4]}_mesh_{idx}.png', orig_img)

            # vis 2d pose
            # tmpkps = np.zeros((3, len(joint_input)))
            # tmpkps[0, :], tmpkps[1, :], tmpkps[2, :] = joint_input[:, 0], joint_input[:, 1], 1
            # swap pevlis and thorax
            # tmpkps[:,-1], tmpkps[:,-2] = tmpkps[:,-2].copy(), tmpkps[:,-1].copy()
            # pose_vis_img = vis_coco_skeleton(pose_vis_img, tmpkps, vis_skeleton, color)#s[idx])
            # cv2.imwrite(output_path + f'{img_name[:-4]}_pose2d_{idx}.png', pose_vis_img)

            # save_obj(out['mesh'], mesh_model.face, output_path + f'{img_name[:-4]}_mesh_{idx}.obj')