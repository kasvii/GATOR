import os
import argparse # python命令行解析库
import torch
import __init_path
import shutil
import random # 新增
import numpy as np

from funcs_utils import save_checkpoint, save_plot, check_data_pararell, count_parameters
from core.config import cfg, update_config

parser = argparse.ArgumentParser(description='Train Pose2Mesh')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123') # 设置随机种子 （似乎没用）
parser.add_argument('--resume_training', action='store_true', help='Resume Training') # 带 --resume_training的时候是true 否则为false
parser.add_argument('--debug', action='store_true', help='reduce dataset items')      # 带 --debug的时候是true 否则为false
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat') # 必要！设置使用的gpu 可以多gpu 中间用逗号隔开
parser.add_argument('--cfg', type=str, help='experiment configure file name') # 必要！配置文件路径

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg) # 导入配置文件

# 设置torch random np的随机种子
torch.manual_seed(args.seed) 
random.seed(args.seed)
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) # 设置使用的GPU
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Trainer, Tester # , LiftTrainer, LiftTester

output_model_dir = os.path.join(cfg.checkpoint_dir, 'Graphormer.py')
output_base_dir = os.path.join(cfg.checkpoint_dir, 'base.py')
output_module_dir = os.path.join(cfg.checkpoint_dir, 'modules.py')
shutil.copyfile(src="./lib/models/Graphormer.py", dst=output_model_dir) # 保存Graphormer.py文件
shutil.copyfile(src="./lib/core/base.py", dst=output_base_dir)          # 保存base.py文件
shutil.copyfile(src="./lib/models/backbones/modules.py", dst=output_module_dir) # 保存modules.py文件

if cfg.MODEL.name == 'pose2mesh_net':
    trainer = Trainer(args, load_dir='') # 端到端训练和测试
    tester = Tester(args)  # if not args.debug else None
# elif cfg.MODEL.name == 'posenet':
#     trainer = LiftTrainer(args, load_dir='') # 2d pose到3d pose训练 （之前方法遗留 目前已经没有这部分）
#     tester = LiftTester(args)  # if not args.debug else None

print("===> Start training...")
for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    trainer.train(epoch)
    trainer.lr_scheduler.step()

    tester.test(epoch, current_model=trainer.model)

    if epoch > 1:
        is_best = tester.joint_error < min(trainer.error_history['joint'])
    else:
        is_best = None

    trainer.error_history['surface'].append(tester.surface_error)
    trainer.error_history['joint'].append(tester.joint_error)

    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': check_data_pararell(trainer.model.state_dict()),
        'optim_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'train_log': trainer.loss_history,
        'test_log': trainer.error_history
    }, epoch, is_best)

    save_plot(trainer.loss_history, epoch)
    save_plot(trainer.error_history['surface'], epoch, title='Surface Error')
    save_plot(trainer.error_history['joint'], epoch, title='Joint Error')

print('Training Finished! All logs were saved in ', cfg.checkpoint_dir)






