import os
import argparse
import torch
import __init_path
import shutil

from funcs_utils import save_checkpoint, save_plot, check_data_pararell, count_parameters
from core.config import cfg, update_config
'wandb'

parser = argparse.ArgumentParser(description='Train Pose2Mesh')

parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume_training', action='store_true', help='Resume Training')
parser.add_argument('--debug', action='store_true', help='reduce dataset items')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--cfg', type=str, help='experiment configure file name')


args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
print('Seed = ', args.seed)
torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

from core.base import Trainer, Tester, LiftTrainer, LiftTester

output_model_dir = os.path.join(cfg.checkpoint_dir, 'Graphormer.py')
output_base_dir = os.path.join(cfg.checkpoint_dir, 'base.py')
output_module_dir = os.path.join(cfg.checkpoint_dir, 'modules.py')
output_posenet_dir = os.path.join(cfg.checkpoint_dir, 'posenet.py')
output_meshnet_dir = os.path.join(cfg.checkpoint_dir, 'meshnet.py')
output_cfg_dir = os.path.join(cfg.checkpoint_dir, 'cfg.yml')
shutil.copyfile(src="./lib/models/Graphormer.py", dst=output_model_dir)
shutil.copyfile(src="./lib/core/base.py", dst=output_base_dir)
shutil.copyfile(src="./lib/models/backbones/modules.py", dst=output_module_dir)
shutil.copyfile(src="./lib/models/posenet.py", dst=output_posenet_dir)
shutil.copyfile(src="./lib/models/meshnet.py", dst=output_meshnet_dir)
shutil.copyfile(src=args.cfg, dst=output_cfg_dir)



if cfg.MODEL.name == 'pose2mesh_net':
    trainer = Trainer(args, load_dir='./experiment/exp_08-28_11_36/checkpoint/best.pth.tar')
    tester = Tester(args)  # if not args.debug else None
elif cfg.MODEL.name == 'posenet':
    trainer = LiftTrainer(args, load_dir='./experiment/exp_09-06_01_19/checkpoint/best.pth.tar')
    tester = LiftTester(args)  # if not args.debug else None

print("===> Start training...")
epoch = cfg.TRAIN.begin_epoch
tester.test(epoch, current_model=trainer.model)
for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    # tester.test(epoch, current_model=trainer.model)
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
        'model_state_dict': trainer.model.state_dict(),   # check_data_pararell(
        'optim_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'train_log': trainer.loss_history,
        'test_log': trainer.error_history
    }, epoch, is_best)

    # save_plot(trainer.loss_history, epoch)
    # save_plot(trainer.error_history['surface'], epoch, title='Surface Error')
    # save_plot(trainer.error_history['joint'], epoch, title='Joint Error')

print('Training Finished! All logs were saved in ', cfg.checkpoint_dir)






