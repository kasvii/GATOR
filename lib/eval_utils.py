# Some functions are borrowed from https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
# Adhere to their licence to use these functions
from pathlib import Path

import torch
import numpy as np
from matplotlib import pyplot as plt


def plot_accel(joints_pred, joints_gt, out_dir, name='', other_preds=['./plot/vibe_accel_pred_val_', './plot/tcmr_accel_pred_val', './plot/maed_accel_pred_val_', './plot/meva_accel_pred_val_']): # './plot/meva_accel_pred_val.npy', 
    joints_pred = joints_pred * 0.001
    joints_gt = joints_gt * 0.001
    name = name[:-1] + '_' + name[-1] + '.npy'
    time = np.arange(len(joints_gt)-2)
    # (N-2)
    print("# of time step: ", len(time))
    plt.figure(figsize=(15, 8))

    accel_gt = compute_accel(joints_gt)

    if other_preds:
        meva_path = other_preds[3] + name
        accel_meva = np.load(meva_path)
        if len(accel_meva) <=2 :
            print('meva ' + name + 'is None')
            return 
        time_meva = np.arange(len(accel_meva)-2) 
        accel_meva = compute_accel(accel_meva)
        
        maed_path = other_preds[2] + name
        accel_maed = np.load(maed_path)
        if len(accel_maed) <=2 :
            print('maed ' + name + 'is None')
            return 
        time_maed = np.arange(len(accel_maed)-2) 
        accel_maed = compute_accel(accel_maed)
        
        tcmr_path = other_preds[1] + name
        accel_tcmr = np.load(tcmr_path, allow_pickle=True)
        accel_tcmr = accel_tcmr.item()
        accel_tcmr = accel_tcmr['kps_3d']
        if len(accel_tcmr) <=2 :
            print('tcmr ' + name + 'is None')
            return            
        time_tcmr = np.arange(len(accel_tcmr)-2) 
        accel_tcmr = compute_accel(accel_tcmr)  # m
        
        
        vide_path = other_preds[0] + name
        accel_vibe = np.load(vide_path)
        if len(accel_vibe) <=2 :
            print('vibe ' + name + 'is None')
            return 
        time_vibe = np.arange(len(accel_vibe)-2) 
        accel_vibe = compute_accel(accel_vibe)
        
        time_len = min(min(min(len(time), len(time_maed)), min(len(time_tcmr), len(time_vibe))), len(time_meva))
        time = time[:time_len]
        accel_tcmr = accel_tcmr[:time_len]
        accel_vibe = accel_vibe[:time_len]
        accel_maed = accel_maed[:time_len]
        accel_meva = accel_meva[:time_len]
        accel_gt = accel_gt[:time_len]
        
        accel_tcmr = np.abs(accel_tcmr - accel_gt)
        # plt.plot(time, accel_tcmr * 1000, label='TCMR', color='#3183F7')#75F74F '')
        accel_vibe = np.abs(accel_vibe - accel_gt)
        plt.plot(time, accel_vibe * 1000, label='VIBE', color='#65D491')
        accel_maed = np.abs(accel_maed - accel_gt)
        plt.plot(time, accel_maed * 1000, label='MAED', color='#D95175')
        accel_meva = np.abs(accel_meva - accel_gt)
        plt.plot(time, accel_meva * 1000, label='MAED', color='#3183F7')
        

    accel_pred = compute_accel(joints_pred)[:time_len]
    accel_err = np.abs(accel_pred - accel_gt)
    plt.plot(time, accel_err*1000, label='Ours', color='#FF7F0E') #

    plt.xlabel('time step', fontsize=10)
    plt.ylabel('acceleration error ($mm/s^2$)', fontsize=10)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )  
    plt.yticks(fontsize=7)
    # plt.grid(color='#303030', linestyle='--', linewidth=0.5, axis='y')
    plt.xlim(-10, len(accel_gt)+10)
    plt.ylim(bottom=-3)

    out_plot_dir = f'./{out_dir}/plot'
    Path(out_plot_dir).mkdir(parents=True, exist_ok=True)
    plot_name = f'./{out_plot_dir}/tcmr_accel_pred_error_{name}.png'
    print("...save plot to ", plot_name)
    plt.savefig(plot_name, bbox_inches = 'tight')
    # np.save(f'./{out_plot_dir}/tcmr_accel_pred_{name}', accel_pred)

def plot_accel_ori(joints_pred, joints_gt, out_dir, name='', other_preds=['./plot/meva_accel_pred_val.npy', './plot/vibe_accel_pred_val.npy']):
    time = np.arange(len(joints_gt)-2)
    # (N-2)
    print("# of time step: ", len(time))
    plt.figure(figsize=(15, 8))

    accel_gt = compute_accel(joints_gt)

    if False and other_preds:
        accel_vibe = np.load(other_preds[1])[:len(time)]
        accel_vibe = np.abs(accel_vibe - accel_gt)
        plt.plot(time, accel_vibe * 1000, label='tcmr', color='#65D491')
        accel_meva = np.load(other_preds[0])[:len(time)]
        accel_meva = np.abs(accel_meva - accel_gt)
        plt.plot(time, accel_meva * 1000, label='MEVA', color='#3183F7')#75F74F '')

    accel_pred = compute_accel(joints_pred)
    accel_err = np.abs(accel_pred - accel_gt)
    plt.plot(time, accel_err, label='TCMR (Ours)', color='#FF7F0E') #

    plt.xlabel('time step', fontsize=10)
    plt.ylabel('acceleration error ($mm/s^2$)', fontsize=10)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )  
    plt.yticks(fontsize=7)
    # plt.grid(color='#303030', linestyle='--', linewidth=0.5, axis='y')
    plt.xlim(-10, len(accel_gt)+10)
    plt.ylim(bottom=-3)

    out_plot_dir = f'./{out_dir}/plot'
    Path(out_plot_dir).mkdir(parents=True, exist_ok=True)
    plot_name = f'./{out_plot_dir}/tcmr_accel_pred_error_{name}.png'
    print("...save plot to ", plot_name)
    plt.savefig(plot_name, bbox_inches = 'tight')
    np.save(f'./{out_plot_dir}/tcmr_accel_pred_{name}', accel_pred)


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_error_verts(pred_verts, target_verts=None, target_theta=None):
    """
    Computes MPJPE over 6890 surface vertices.
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    if target_verts is None:
        from lib.models.smpl import SMPL_MODEL_DIR
        from lib.models.smpl import SMPL
        device = 'cpu'
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1, # target_theta.shape[0],
        ).to(device)

        betas = torch.from_numpy(target_theta[:,75:]).to(device)
        pose = torch.from_numpy(target_theta[:,3:75]).to(device)

        target_verts = []
        b_ = torch.split(betas, 5000)
        p_ = torch.split(pose, 5000)

        for b,p in zip(b_,p_):
            output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
            target_verts.append(output.vertices.detach().cpu().numpy())

        target_verts = np.concatenate(target_verts, axis=0)

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def align_by_pelvis(joints):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """

    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa
