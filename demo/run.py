import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
    # cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
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

        model_chk_path = './experiment/exp_03-17_20:35/checkpoint/best.pth.tar' # './experiment/exp_03-15_20:51/checkpoint/best.pth.tar' # './experiment/pose2mesh_human36J_train_human36/final.pth.tar'

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
        model_chk_path = './experiment/exp_03-29_22:22/checkpoint/best.pth.tar'# './experiment/exp_03-26_12:35/exp_03-26_12:35/checkpoint/best.pth.tar' # './experiment/pose2mesh_cocoJ_train_human36_coco_muco/final.pth.tar'

    elif joint_category == 'smpl':
        joint_regressor = mesh_model.layer['neutral'].th_J_regressor.numpy().astype(np.float32)
        joint_num = 24
        skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, flip_pairs, levels=9)
        model_chk_path = './experiment/pose2mesh_smplJ_train_surreal/final.pth.tar'

    elif joint_category == 'mano':
        joint_regressor = mesh_model.joint_regressor
        joint_num = 21
        skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        hori_conn = (
        (1, 5), (5, 9), (9, 13), (13, 17), (2, 6), (6, 10), (10, 14), (14, 18), (3, 7), (7, 11), (11, 15), (15, 19),
        (4, 8), (8, 12), (12, 16), (16, 20))
        graph_Adj, graph_L, graph_perm, graph_perm_reverse = \
            build_coarse_graphs(mesh_model.face, joint_num, skeleton, hori_conn, levels=6)
        model_chk_path = './experiment/pose2mesh_manoJ_train_freihand/final.pth.tar'

    else:
        raise NotImplementedError(f"{joint_category}: unknown joint set category")

    # model = models.pose2mesh_net.get_model(joint_num, graph_L)
    joint_regressor = torch.Tensor(joint_regressor)
    model = models.Graphormer.get_model(joint_num, embed_dim=512, depth=6, graph_adj=graph_Adj, GCN_depth=2, J_regressor=joint_regressor)
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
    print('joint_img ', joint_img)
    print('joint_img.shape ', joint_img.shape)
    pred_mesh = model(joint_img)
    print('pred_mesh.shape, ', pred_mesh.shape)
    print('pred_mesh, ', pred_mesh)
    # pred_mesh = pred_mesh[:, graph_perm_reverse[:mesh_model.face.max() + 1], :]
    pred_3d_joint = torch.matmul(joint_regressor, pred_mesh)
    print('pred_mesh.shape, ', pred_mesh.shape)
    print('pred_3d_joint.shape, ', pred_3d_joint.shape)
    print('pred_mesh, ', pred_mesh)
    print('pred_3d_joint, ', pred_3d_joint)

    out = {}
    # assume batch=1
    project_net.train()
    for j in range(0, 1500):
        # print(j,'th turn')
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())

        loss = criterion(pred_2d_joint, target_joint[:, :17, :])
        # print('pred_2d_joint', pred_2d_joint)
        # print('target_joint', target_joint[:, :17, :])
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
        # joint_input = np.load(input_path)
        # 470 joint_input = np.array([[881.0711669921875, 327.9759216308594, 0.9688085317611694], [882.0493774414062, 311.2183532714844, 0.9713721871376038], [866.5530395507812, 310.85394287109375, 0.9840340614318848], [814.5902099609375, 308.91082763671875, 0.6830089092254639], [799.7344970703125, 309.2054748535156, 0.9910727739334106], [812.5479736328125, 389.0264892578125, 0.8218358755111694], [711.561767578125, 419.1875, 0.9023309946060181], [815.2410888671875, 485.3656311035156, 0.8924195766448975], [662.2780151367188, 602.4942016601562, 0.9353774785995483], [889.8814697265625, 577.4931030273438, 0.9343879222869873], [827.8234252929688, 666.554443359375, 0.9116863012313843], [685.0912475585938, 665.275634765625, 0.8286945819854736], [585.05810546875, 671.823974609375, 0.7840192317962646], [772.5535278320312, 859.11572265625, 0.9479916095733643], [533.3143920898438, 920.1845092773438, 0.9543132781982422], [694.680419921875, 1065.3948974609375, 0.9213494658470154], [315.95794677734375, 1056.498046875, 0.9275729656219482]])
        # 765 
        joint_input = np.array([[1291.3638916015625, 133.28578186035156, 0.9982743263244629], [1314.344482421875, 115.85456848144531, 0.9863348007202148], [1280.970458984375, 115.5665283203125, 0.9855356216430664], [1362.3145751953125, 128.57400512695312, 0.9809767007827759], [1271.8909912109375, 122.30479431152344, 0.8529441952705383], [1416.35693359375, 237.69175720214844, 0.941078782081604], [1223.92333984375, 227.0111083984375, 0.9616540670394897], [1463.168212890625, 375.3280334472656, 0.9519579410552979], [1162.106201171875, 343.2330322265625, 0.9806461930274963], [1427.9552001953125, 461.90924072265625, 0.9899344444274902], [1146.637451171875, 450.2824401855469, 0.9859115481376648], [1350.5384521484375, 523.412353515625, 0.8887602686882019], [1244.294677734375, 517.5330200195312, 0.9031262397766113], [1283.4971923828125, 713.2562255859375, 0.8686809539794922], [1293.2506103515625, 721.4976196289062, 0.9003548622131348], [1206.38525390625, 861.4423828125, 0.919001579284668], [1322.1827392578125, 949.37841796875, 0.917795717716217]])
        # "image_00228.jpg", "sequence": "downtown_crossStreets_00"
        # joint_input = np.array([[1014.7752075195312, 349.32080078125, 0.9738960266113281], [1025.5882568359375, 336.6585388183594, 0.9833111763000488], [1002.6685791015625, 338.4422607421875, 0.9769190549850464], [1045.2711181640625, 347.9502868652344, 0.9723900556564331], [985.1000366210938, 350.1415710449219, 0.9729398488998413], [1088.77197265625, 417.8061828613281, 0.9292754530906677], [970.8881225585938, 425.0147399902344, 0.9208110570907593], [1154.833740234375, 519.2015991210938, 0.9896000027656555], [981.7300415039062, 521.2548217773438, 0.8698204159736633], [1114.741455078125, 468.0435485839844, 0.9304746985435486], [952.7391357421875, 477.0352783203125, 0.9356261491775513], [1080.4012451171875, 635.6732788085938, 0.8746290802955627], [995.5950927734375, 634.3587036132812, 0.8925485014915466], [1102.5732421875, 780.8106079101562, 0.9535666704177856], [974.2100219726562, 780.2005004882812, 0.9300762414932251], [1112.82080078125, 898.4293212890625, 0.947628915309906], [944.982421875, 936.1638793945312, 0.8977633714675903]])
        # "image_00601.jpg", "sequence": "downtown_walkBridge_01"   id": 1384
        # joint_input = np.array([[780.7032470703125, 198.1530303955078, 0.9884791970252991], [777.8560791015625, 180.74325561523438, 0.8751495480537415], [771.4534912109375, 179.54153442382812, 0.9747956991195679], [680.1563110351562, 184.4508056640625, 0.7519898414611816], [730.7862548828125, 186.69326782226562, 0.9934664964675903], [640.6572265625, 287.0105285644531, 0.8951035737991333], [751.4580688476562, 310.9392395019531, 0.897413969039917], [601.0811767578125, 379.3297119140625, 0.8647664785385132], [779.7213745117188, 478.7891845703125, 0.8707459568977356], [691.049072265625, 404.87847900390625, 0.30828338861465454], [825.8043212890625, 619.7081298828125, 0.9060667753219604], [668.1966552734375, 588.9525756835938, 0.8041946887969971], [758.3320922851562, 600.6803588867188, 0.7621316909790039], [634.5796508789062, 784.700927734375, 0.8905958533287048], [810.6881713867188, 784.8981323242188, 0.8652622699737549], [542.9175415039062, 972.6380004882812, 0.8776708841323853], [822.8713989257812, 964.57080078125, 0.936269223690033]])
        # "image_01330.jpg", "sequence": "downtown_enterShop_00"    id": 23943
        # joint_input = np.array([[567.8900756835938, 337.86669921875, 0.9735062122344971], [600.2353515625, 323.4501647949219, 1.0071179866790771], [584.3731689453125, 311.68267822265625, 0.9782720804214478], [684.848388671875, 381.61029052734375, 0.9886583089828491], [678.5181884765625, 375.5718994140625, 0.6941081285476685], [667.4010620117188, 606.4439086914062, 0.8883714079856873], [651.9249877929688, 548.4490966796875, 0.8586289882659912], [601.6356201171875, 907.3916015625, 0.9188063740730286], [540.4182739257812, 614.6672973632812, 0.9065395593643188], [483.2955017089844, 1110.161865234375, 0.8909575939178467], [384.5017395019531, 590.236328125, 0.9622728824615479], [670.5775146484375, 1027.14453125, 0.8877759575843811], [612.968505859375, 1012.6332397460938, 0.746532678604126], [684.6774291992188, 1333.1085205078125, 0.8747341632843018], [572.595458984375, 1269.5584716796875, 0.8818594217300415], [793.3528442382812, 1652.1605224609375, 0.9210872650146484], [509.013916015625, 1465.587646484375, 0.9482569694519043]])
        # "image_00975.jpg", "sequence": "downtown_enterShop_00"    id": 23588
        # joint_input = np.array([[418.6974182128906, 387.3379821777344, 0.9742119312286377], [416.6646728515625, 361.3941955566406, 0.8840849995613098], [408.1519470214844, 359.17425537109375, 0.9748662710189819], [323.3238830566406, 365.45916748046875, 0.6693601012229919], [352.50714111328125, 362.9668884277344, 0.9778900742530823], [260.69110107421875, 530.96142578125, 0.8385570049285889], [371.1009826660156, 537.1260375976562, 0.8834738731384277], [375.6231384277344, 557.1842651367188, 0.6312296390533447], [586.70703125, 533.072021484375, 0.9316403865814209], [456.805419921875, 496.45330810546875, 0.8945375084877014], [646.1826171875, 436.7541198730469, 0.9285247325897217], [329.6337585449219, 913.5027465820312, 0.7677141427993774], [389.5152282714844, 928.2778930664062, 0.8181101083755493], [353.9755859375, 1202.777099609375, 0.787614107131958], [371.43756103515625, 1231.2984619140625, 0.889376163482666], [354.6571960449219, 1418.67578125, 0.8767959475517273], [325.892578125, 1497.394775390625, 0.9056479930877686]])
        # "image_01658.jpg", "sequence": "downtown_windowShopping_00"    id": 22323
        # joint_input = np.array([[406.48681640625, 248.04022216796875, 0.9922769069671631], [413.40240478515625, 210.9333038330078, 0.972636342048645], [352.3743896484375, 212.58026123046875, 0.9956743121147156], [412.334228515625, 239.42572021484375, 0.5648905038833618], [227.15382385253906, 260.1181640625, 0.9878902435302734], [305.8746643066406, 507.20184326171875, 0.8538480997085571], [212.85552978515625, 504.3807067871094, 0.8991198539733887], [380.7904968261719, 827.9749145507812, 0.6257473230361938], [384.2125244140625, 873.7959594726562, 0.8970165252685547], [445.94354248046875, 948.447998046875, 0.4021536409854889], [644.7053833007812, 1052.0216064453125, 0.9550034999847412], [341.9842834472656, 1071.51953125, 0.7809898257255554], [241.5035858154297, 1090.8524169921875, 0.799910306930542], [460.93328857421875, 1357.64697265625, 0.8725844025611877], [203.3179931640625, 1481.5625, 0.8870750665664673], [593.0641479492188, 1621.234619140625, 0.9253822565078735], [126.8834228515625, 1824.8160400390625, 0.9229903817176819]])
        # "image_00955.jpg", "sequence": "downtown_stairs_00"    id": 20380
        # joint_input = np.array([[1014.7579956054688, 212.93618774414062, 0.9940717220306396], [1017.945068359375, 206.10231018066406, 0.9844014644622803], [1001.795166015625, 203.2043914794922, 0.9804689884185791], [1011.3636474609375, 221.2775421142578, 0.6468698382377625], [969.4489135742188, 213.9368133544922, 0.9829534292221069], [993.7393188476562, 281.1324157714844, 0.8576537370681763], [957.1396484375, 266.90179443359375, 0.9347310066223145], [997.9892578125, 377.1578674316406, 0.6606991291046143], [923.431396484375, 350.70574951171875, 0.9400666952133179], [1025.18505859375, 452.13201904296875, 0.8489983081817627], [937.3347778320312, 435.1717834472656, 0.87996506690979], [1010.2379150390625, 461.268310546875, 0.8396525382995605], [978.9419555664062, 459.0256652832031, 0.8845096230506897], [1001.3433227539062, 594.3956298828125, 0.8643112182617188], [991.89501953125, 593.8125610351562, 0.88877272605896], [923.6798706054688, 683.2339477539062, 0.7630109786987305], [1000.2630615234375, 723.5374145507812, 0.75624018907547]])
        
        # "image_00228.jpg", "sequence": "downtown_crossStreets_00"    id": 10117
        # joint_input = np.array([[1303.8065185546875, 413.61474609375, 0.9464343190193176], [1317.40283203125, 398.52386474609375, 0.9655452966690063], [1291.5118408203125, 402.14288330078125, 0.9429111480712891], [1351.1846923828125, 390.4922180175781, 0.976535439491272], [1280.51953125, 404.697021484375, 0.8987163305282593], [1403.9361572265625, 479.3185729980469, 0.9151264429092407], [1272.6614990234375, 493.75250244140625, 0.9178928732872009], [1432.6373291015625, 595.5546875, 0.9596275687217712], [1258.492431640625, 599.484375, 0.9617007970809937], [1397.40087890625, 692.085205078125, 0.9500788450241089], [1248.2008056640625, 690.1399536132812, 0.961195707321167], [1385.65380859375, 701.5848999023438, 0.8390904068946838], [1293.306640625, 706.6954956054688, 0.8359686732292175], [1414.003662109375, 839.0882568359375, 0.9303622841835022], [1281.481689453125, 851.074951171875, 0.9220573902130127], [1430.217529296875, 938.3287963867188, 0.9157471060752869], [1287.853271484375, 1005.8385009765625, 0.9180651307106018]])
        # 470
        # joint_input = np.array([[870.6383666992188, 447.98834228515625, 0.962871789932251], [883.7838745117188, 435.6572570800781, 0.950567901134491], [870.9426879882812, 435.09552001953125, 0.8925950527191162], [933.2949829101562, 437.2872314453125, 0.9734932780265808], [927.3203125, 437.79864501953125, 0.5632604360580444], [1037.57275390625, 530.8296508789062, 0.8607637882232666], [874.6785278320312, 513.3493041992188, 0.873233437538147], [1115.6312255859375, 643.4410400390625, 0.9444265365600586], [838.6197509765625, 605.6255493164062, 0.9346917867660522], [1174.035400390625, 746.66259765625, 0.9443399906158447], [855.496337890625, 682.7369995117188, 0.881961464881897], [1042.3951416015625, 717.5360717773438, 0.8516509532928467], [938.648193359375, 725.0814819335938, 0.8152158260345459], [1118.63525390625, 847.6870727539062, 0.9264196157455444], [946.7897338867188, 872.9267578125, 0.8959358334541321], [1158.4959716796875, 958.441162109375, 0.8958034515380859], [1078.1263427734375, 1008.8331298828125, 0.909757137298584]])
        # 765
        # joint_input = np.array([[663.7183837890625, 341.7447509765625, 0.9823532700538635], [667.3895874023438, 333.1610107421875, 0.9746720194816589], [645.5592651367188, 332.0917053222656, 0.9989058971405029], [663.621337890625, 348.1236572265625, 0.5124022960662842], [599.947265625, 351.15496826171875, 0.9654830694198608], [641.8267211914062, 423.2384948730469, 0.9108327627182007], [582.9623413085938, 429.0538024902344, 0.8924267292022705], [706.7821655273438, 493.0632019042969, 0.7673307657241821], [651.1796875, 519.138671875, 0.9277793169021606], [793.5637817382812, 502.9394226074219, 0.5441487431526184], [750.8927612304688, 492.5293884277344, 0.9832832217216492], [627.8526611328125, 620.4685668945312, 0.7229486703872681], [580.0308227539062, 638.513671875, 0.7528864145278931], [755.224609375, 653.346923828125, 0.8190845251083374], [744.7430419921875, 680.5631103515625, 0.8798133730888367], [676.186279296875, 796.0271606445312, 0.7185136079788208], [647.3142700195312, 816.0200805664062, 0.8374062776565552]])
        # 000
        # joint_input = np.array([[1046.8380126953125, 323.2848205566406, 0.9850720167160034], [1061.265625, 312.31304931640625, 0.971976101398468], [1034.957763671875, 312.1553955078125, 0.9742059707641602], [1085.9005126953125, 320.5824890136719, 0.9653753042221069], [1022.3115234375, 321.04278564453125, 0.9671025276184082], [1119.519287109375, 411.29498291015625, 0.9364444613456726], [989.388671875, 410.2996520996094, 0.9497737884521484], [1132.259521484375, 514.4891357421875, 0.9495478868484497], [974.6029663085938, 516.8375244140625, 0.9678980112075806], [1124.655517578125, 611.6096801757812, 0.9497033357620239], [969.0530395507812, 600.7280883789062, 0.9545535445213318], [1087.58154296875, 618.6626586914062, 0.8378478288650513], [1009.4158325195312, 617.4929809570312, 0.8175421357154846], [1076.6781005859375, 761.1249389648438, 0.9038445949554443], [1023.513427734375, 760.8306884765625, 0.878867506980896], [1068.8226318359375, 890.8050537109375, 0.9113110303878784], [1033.46533203125, 886.0186767578125, 0.9380940794944763]])
        '''
        joint_input = np.array([
            327.8859558105469,
            168.5027618408203,
            0.6776717305183411,
            325.5813903808594,
            168.5027618408203,
            0.8455424308776855,
            332.4950866699219,
            168.5027618408203,
            0.7469667792320251,
            323.27685546875,
            182.33010864257812,
            0.9624876379966736,
            346.3224182128906,
            182.33010864257812,
            0.8831964731216431,
            314.0586242675781,
            212.2893524169922,
            0.9697844982147217,
            353.236083984375,
            191.54833984375,
            0.8907769322395325,
            325.5813903808594,
            212.2893524169922,
            0.45144981145858765,
            348.6269836425781,
            138.54351806640625,
            0.9556221961975098,
            344.0178527832031,
            193.85289001464844,
            0.3825254440307617,
            332.4950866699219,
            99.36604309082031,
            0.9363729953765869,
            330.1905212402344,
            276.81695556640625,
            0.8879741430282593,
            357.84521484375,
            274.51239013671875,
            0.8798465728759766,
            304.84039306640625,
            329.82177734375,
            0.9158643484115601,
            350.9315185546875,
            343.6491394042969,
            0.9301739931106567,
            284.099365234375,
            387.43572998046875,
            0.937041163444519,
            348.6269836425781,
            408.1767272949219,
            0.9343369603157043
        ])
        '''
        joint_input = joint_input.reshape(17,3)
        print(joint_input)
        print(joint_input.shape)
        if joint_set == 'coco':
            joint_input = add_pelvis(joint_input, coco_joints_name)
            joint_input = add_neck(joint_input, coco_joints_name)
            joint_input = joint_input[:,:2]
        print('joint_input.shape, ', joint_input.shape)
        print('joint_input, ', joint_input)

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

'''
# coco 872
[
            213.36602783203125,
            186.65708923339844,
            0.9548212289810181,
            219.60821533203125,
            177.29380798339844,
            0.9617825150489807,
            213.36602783203125,
            177.29380798339844,
            0.9182882308959961,
            244.57696533203125,
            180.41490173339844,
            0.9524273872375488,
            247.69805908203125,
            177.29380798339844,
            0.6026554703712463,
            282.03009033203125,
            227.23130798339844,
            0.820064127445221,
            244.57696533203125,
            230.35240173339844,
            0.8331472873687744,
            325.72540283203125,
            295.8953857421875,
            0.9398249387741089,
            241.45587158203125,
            289.6531982421875,
            0.8335295915603638,
            291.39337158203125,
            358.3172607421875,
            0.8627396821975708,
            194.63946533203125,
            333.3485107421875,
            0.852901816368103,
            303.87774658203125,
            377.0438232421875,
            0.7198745012283325,
            250.81915283203125,
            373.9227294921875,
            0.7650312185287476,
            294.51446533203125,
            480.0399169921875,
            0.7750300168991089,
            222.72930908203125,
            455.0711669921875,
            0.7642873525619507,
            375.66290283203125,
            417.6180419921875,
            0.8532792925834656,
            247.69805908203125,
            576.7938232421875,
            0.35144901275634766
        ]

# coco785
[
            368.26007080078125,
            79.16297149658203,
            0.9674993753433228,
            375.0116271972656,
            74.66192626953125,
            0.9745193123817444,
            361.5085144042969,
            74.66192626953125,
            0.9654536843299866,
            386.26422119140625,
            76.9124526977539,
            0.9630154371261597,
            357.0074462890625,
            79.16297149658203,
            0.965564489364624,
            408.7694396972656,
            112.92078399658203,
            0.8827023506164551,
            363.759033203125,
            128.6744384765625,
            0.9143320918083191,
            444.77777099609375,
            151.1796417236328,
            0.6681779623031616,
            336.7527770996094,
            162.4322509765625,
            0.9492111802101135,
            447.0282897949219,
            162.4322509765625,
            0.9055604934692383,
            305.2454833984375,
            178.18589782714844,
            0.9261032342910767,
            429.0241394042969,
            211.94371032714844,
            0.8438689708709717,
            402.01788330078125,
            218.6952667236328,
            0.862857460975647,
            433.5251770019531,
            288.46142578125,
            0.9063125252723694,
            372.7611083984375,
            272.707763671875,
            0.9426339864730835,
            476.2850646972656,
            360.47808837890625,
            0.881801962852478,
            404.2684020996094,
            342.4739074707031,
            0.8836283087730408
        ]

coco39551
[
            267.6304931640625,
            91.34892272949219,
            0.984908401966095,
            267.6304931640625,
            85.8086929321289,
            0.9407845735549927,
            263.9369812011719,
            83.96194458007812,
            0.9609849452972412,
            238.0825653076172,
            80.2684555053711,
            0.7636619806289673,
            243.622802734375,
            82.11520385742188,
            0.9860527515411377,
            204.84115600585938,
            85.8086929321289,
            0.9473680257797241,
            249.1630401611328,
            115.35660552978516,
            0.920359194278717,
            158.67254638671875,
            96.88916015625,
            0.9522513747215271,
            219.6151123046875,
            155.9849853515625,
            0.6824935674667358,
            149.43881225585938,
            133.8240509033203,
            0.9413890838623047,
            162.36602783203125,
            139.36428833007812,
            0.4807674288749695,
            166.0595245361328,
            167.06546020507812,
            0.8138391971588135,
            195.60743713378906,
            168.91220092773438,
            0.7791983485221863,
            201.14767456054688,
            235.39501953125,
            0.963543713092804,
            260.2434997558594,
            231.70152282714844,
            0.9525140523910522,
            140.20509338378906,
            285.2571105957031,
            0.8814355731010437,
            278.7109375,
            312.9582824707031,
            0.8973631262779236
        ]

# coco158227
[
            263.5737609863281,
            94.23517608642578,
            0.9735842943191528,
            265.8665466308594,
            80.47840881347656,
            0.9485635757446289,
            245.2313995361328,
            87.3567886352539,
            0.9506821632385254,
            275.0377502441406,
            82.77120208740234,
            0.6501384377479553,
            208.5466766357422,
            110.28474426269531,
            0.9258067607879639,
            284.20892333984375,
            153.8478546142578,
            0.7967522144317627,
            197.08270263671875,
            190.53257751464844,
            0.8271675705909729,
            226.8890380859375,
            211.167724609375,
            0.8472625613212585,
            199.37550354003906,
            286.8299865722656,
            0.9100869297981262,
            148.9340057373047,
            238.6812744140625,
            0.858493447303772,
            148.9340057373047,
            247.85244750976562,
            0.8895348310470581,
            336.9432067871094,
            339.56427001953125,
            0.752694308757782,
            258.9881896972656,
            351.0282287597656,
            0.8195117115974426,
            334.6504211425781,
            302.8795471191406,
            0.01348206028342247,
            183.325927734375,
            296.00115966796875,
            0.25658145546913147,
            336.9432067871094,
            337.2714538574219,
            0.10526910424232483,
            215.42506408691406,
            302.8795471191406,
            0.04103358834981918
        ]

# coco163118
[
            227.85719299316406,
            132.55343627929688,
            0.9804895520210266,
            223.96343994140625,
            132.55343627929688,
            0.8984766006469727,
            223.96343994140625,
            134.50030517578125,
            0.9458169937133789,
            220.06968688964844,
            150.07530212402344,
            0.6321744918823242,
            225.9103240966797,
            150.07530212402344,
            0.9806336164474487,
            216.1759490966797,
            173.43780517578125,
            0.7973304986953735,
            239.53843688964844,
            157.86280822753906,
            0.907429575920105,
            216.1759490966797,
            204.5878143310547,
            0.5942553281784058,
            259.0072021484375,
            132.55343627929688,
            0.9175241589546204,
            237.59156799316406,
            241.57843017578125,
            0.8407312631607056,
            272.63531494140625,
            97.50968170166016,
            0.9409424066543579,
            208.38844299316406,
            227.9503173828125,
            0.7987594604492188,
            225.9103240966797,
            233.7909393310547,
            0.8332584500312805,
            190.86656188964844,
            288.3034362792969,
            0.9881007671356201,
            202.54782104492188,
            292.1971740722656,
            0.9686388373374939,
            169.45094299316406,
            340.8690490722656,
            0.8368459939956665,
            167.5040740966797,
            338.92218017578125,
            0.8998116254806519
        ]

# coco 162581
[
            212.94293212890625,
            191.81690979003906,
            0.9630038142204285,
            222.5799102783203,
            189.4076690673828,
            0.9597446918487549,
            215.3521728515625,
            186.99842834472656,
            0.886282205581665,
            241.85386657714844,
            196.63540649414062,
            0.9844306707382202,
            241.85386657714844,
            194.22616577148438,
            0.6930044889450073,
            224.98916625976562,
            223.13710021972656,
            0.8900282979011536,
            251.4908447265625,
            203.86314392089844,
            0.8618037104606628,
            179.21351623535156,
            201.45388793945312,
            0.9627401828765869,
            203.3059539794922,
            201.45388793945312,
            0.6898422241210938,
            164.75804138183594,
            160.49673461914062,
            0.892442524433136,
            167.1672821044922,
            172.54295349121094,
            0.7407011985778809,
            270.7648010253906,
            319.50689697265625,
            0.8323447108268738,
            239.4446258544922,
            312.2791442871094,
            0.757514238357544,
            268.3555603027344,
            399.011962890625,
            0.941317081451416,
            220.17066955566406,
            379.7380065917969,
            0.9271687269210815,
            311.72198486328125,
            420.6951599121094,
            0.9303296804428101,
            212.94293212890625,
            473.69854736328125,
            0.8955007195472717
        ]

# coco 229553
[
            387.21173095703125,
            125.97547149658203,
            0.9602410197257996,
            398.8039245605469,
            114.38328552246094,
            0.9766139984130859,
            375.6195373535156,
            114.38328552246094,
            0.9821175336837769,
            414.2601623535156,
            122.11141204833984,
            0.9602903127670288,
            352.4351806640625,
            114.38328552246094,
            0.9671415090560913,
            406.53204345703125,
            183.93641662597656,
            0.9404184818267822,
            321.5226745605469,
            156.88796997070312,
            0.9361954927444458,
            414.2601623535156,
            195.52859497070312,
            0.3233306407928467,
            221.05703735351562,
            180.0723419189453,
            0.9627863168716431,
            410.3961181640625,
            191.66453552246094,
            0.3141307234764099,
            128.31954956054688,
            183.93641662597656,
            0.947207510471344,
            387.21173095703125,
            319.1785888671875,
            0.8230726718902588,
            321.5226745605469,
            303.72235107421875,
            0.7765616178512573,
            476.0851745605469,
            199.3926544189453,
            0.9250184893608093,
            391.0758056640625,
            222.57704162597656,
            0.9706646203994751,
            445.17266845703125,
            334.6348571777344,
            0.8849520683288574,
            317.6585998535156,
            342.36297607421875,
            0.9082498550415039
        ]

# coco345466
[
            247.48509216308594,
            69.93343353271484,
            0.9003713130950928,
            247.48509216308594,
            67.39488983154297,
            0.7644938230514526,
            247.48509216308594,
            67.39488983154297,
            0.9725913405418396,
            217.02259826660156,
            77.54905700683594,
            0.9590156674385071,
            242.40802001953125,
            72.47197723388672,
            0.972507119178772,
            217.02259826660156,
            113.0886459350586,
            0.9068365097045898,
            262.7163391113281,
            90.24176788330078,
            0.9541717171669006,
            194.17572021484375,
            128.3199005126953,
            0.9625438451766968,
            310.9486389160156,
            90.24176788330078,
            0.9970510005950928,
            148.48196411132812,
            115.62718200683594,
            0.9504982829093933,
            316.0257263183594,
            113.0886459350586,
            0.9874037504196167,
            232.25384521484375,
            194.3219757080078,
            0.9112695455551147,
            275.4090576171875,
            191.78343200683594,
            0.8600417375564575,
            194.17572021484375,
            252.70843505859375,
            0.954574704170227,
            333.7955322265625,
            227.32302856445312,
            0.9516899585723877,
            143.40489196777344,
            285.70947265625,
            0.9223372340202332,
            361.719482421875,
            298.4021911621094,
            0.9206430315971375
        ]

## coco406417 0
[
            263.4304504394531,
            299.12896728515625,
            0.9739179611206055,
            270.6910095214844,
            291.868408203125,
            0.9820672273635864,
            256.169921875,
            289.4482421875,
            0.9784958362579346,
            282.79193115234375,
            294.2886047363281,
            0.953694224357605,
            244.06900024414062,
            289.4482421875,
            0.9565865993499756,
            292.47265625,
            325.7509765625,
            0.937785267829895,
            236.80845642089844,
            330.5913391113281,
            0.9296268224716187,
            326.3551940917969,
            362.0537109375,
            0.9734171628952026,
            219.8671875,
            376.5747985839844,
            0.9539111852645874,
            338.45611572265625,
            408.03717041015625,
            0.9696297645568848,
            183.564453125,
            405.6169738769531,
            0.9642019271850586,
            282.79193115234375,
            434.6591796875,
            0.8814417123794556,
            241.64881896972656,
            439.4995422363281,
            0.8558813333511353,
            302.15338134765625,
            531.4664306640625,
            0.9068244695663452,
            248.90936279296875,
            531.4664306640625,
            0.9268237352371216,
            316.6744689941406,
            628.2737426757812,
            0.9340817332267761,
            258.590087890625,
            621.01318359375,
            0.9204174280166626
        ]

coco 406417 1
[
            429.49517822265625,
            310.9129333496094,
            0.9700465202331543,
            434.2386474609375,
            303.7976989746094,
            0.9741040468215942,
            420.0081787109375,
            306.16943359375,
            0.9865838885307312,
            453.2126159667969,
            303.7976989746094,
            0.9640632271766663,
            417.6364440917969,
            313.28466796875,
            0.960479736328125,
            467.4430847167969,
            353.6043395996094,
            0.8777365684509277,
            422.37994384765625,
            351.23260498046875,
            0.9285745024681091,
            462.6995849609375,
            405.78271484375,
            0.49705737829208374,
            412.8929443359375,
            405.78271484375,
            0.9515007138252258,
            403.4059753417969,
            460.3328552246094,
            0.3182479739189148,
            403.4059753417969,
            460.3328552246094,
            0.9459711313247681,
            486.41705322265625,
            460.3328552246094,
            0.86888587474823,
            453.2126159667969,
            457.96112060546875,
            0.8388861417770386,
            500.64752197265625,
            552.8309326171875,
            0.933769702911377,
            424.7516784667969,
            552.8309326171875,
            0.8944059610366821,
            469.8148193359375,
            633.47021484375,
            0.9270962476730347,
            386.80377197265625,
            633.47021484375,
            0.9449597597122192
        ]
# coco 406417 2
[
            43.84914016723633,
            334.1169738769531,
            0.9725020527839661,
            46.1025276184082,
            322.85003662109375,
            0.9741891026496887,
            37.088985443115234,
            331.86358642578125,
            0.9776477813720703,
            66.38299560546875,
            313.83648681640625,
            0.9552983045578003,
            37.088985443115234,
            336.3703308105469,
            0.663804292678833,
            104.69054412841797,
            349.8906555175781,
            0.9321939945220947,
            52.8626823425293,
            370.1711120605469,
            0.9261319637298584,
            111.45069885253906,
            403.9718933105469,
            0.42770105600357056,
            50.60929870605469,
            428.7591552734375,
            0.9334491491317749,
            32.58221435546875,
            487.34716796875,
            0.3956500291824341,
            32.58221435546875,
            487.34716796875,
            0.9371311068534851,
            131.73117065429688,
            467.0666809082031,
            0.8135204315185547,
            91.17023468017578,
            469.320068359375,
            0.8325495719909668,
            120.46424102783203,
            566.2156372070312,
            0.9149391055107117,
            70.88976287841797,
            566.2156372070312,
            0.8713293075561523,
            113.70408630371094,
            633.8171997070312,
            0.9242180585861206,
            59.62283706665039,
            633.8171997070312,
            0.9488473534584045
        ]

# coco 409475 0
[
            195.06578063964844,
            183.3794403076172,
            0.9709272384643555,
            201.53668212890625,
            174.7515869140625,
            0.9845074415206909,
            186.4379119873047,
            176.90855407714844,
            0.9737989902496338,
            212.32151794433594,
            174.7515869140625,
            0.9675049185752869,
            177.81005859375,
            183.3794403076172,
            0.9587695002555847,
            233.89117431640625,
            222.204833984375,
            0.9307504892349243,
            167.0252227783203,
            224.36180114746094,
            0.9191893339157104,
            236.0481414794922,
            269.6580810546875,
            0.9301039576530457,
            149.76950073242188,
            276.1289978027344,
            0.9300088286399841,
            274.87353515625,
            293.38470458984375,
            0.9362261295318604,
            164.86825561523438,
            289.07080078125,
            0.9578052759170532,
            218.7924041748047,
            325.73919677734375,
            0.8679170608520508,
            179.96702575683594,
            327.89617919921875,
            0.8321192264556885,
            216.63543701171875,
            407.70391845703125,
            0.9034627676010132,
            186.4379119873047,
            405.5469665527344,
            0.9194183349609375,
            201.53668212890625,
            468.0989685058594,
            0.8927976489067078,
            158.39735412597656,
            465.9420166015625,
            0.8727925419807434
        ]
'''