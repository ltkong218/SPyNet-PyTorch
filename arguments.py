import argparse
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch SPyNet Training')

# general options
parser.add_argument('--mode', default='train', type=str, metavar='MODE', help='working mode, select train or test')
parser.add_argument('--manual_seed', default=1, type=int, metavar='N', help='manual seed')
parser.add_argument('--net_gpu_id', default=0, type=int, metavar='N', help='CNN network gpu id')
parser.add_argument('--aug_make_data_gpu_id', default=0, type=int, metavar='N', help='data augmentation and making gpu id')
parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of data loading and processing workers')
parser.add_argument('--use_visdom', default=False, type=str2bool, help='use visdom to visualize training loss and testing loss')
parser.add_argument('--use_pytorch_model', default=True, type=str2bool, help='use pytorch or torch models')

# parallel options
parser.add_argument('--is_parallel', default=False, type=str2bool, help='data parallel option')
parser.add_argument('--parallel_gpu_ids', default='1, 2, 3', metavar='LIST', help='list of gpu ids for data parallel')

# data options
parser.add_argument('--FineHeight', default=384, type=int, metavar='H', help='data augmentation height in training')
parser.add_argument('--FineWidth', default=512, type=int, metavar='W', help='data augmentation width in training')
parser.add_argument('--level', default=1, type=int, metavar='L', help='training level')

# path options
parser.add_argument('--flyingchairs_dataset_path', default='/home/ltkong/Datasets/FlyingChairs/FlyingChairs_release/data', type=str, metavar='DIR', help='FlyingChairs dataset path')
parser.add_argument('--mpisintel_dataset_path', default='/home/ltkong/Datasets/MPI-Sintel', type=str, metavar='DIR', help='MPI-Sintel dataset path')
parser.add_argument('--checkpoint_path', default='./checkpoints', type=str, metavar='DIR', help='checkpoint path')
parser.add_argument('--model_path', default='./models/myClean', type=str, metavar='DIR', help='model path')
parser.add_argument('--result_path', default='./results', type=str, metavar='DIR', help='results path')
parser.add_argument('--FlyingChairs_train_val_path', default='./FlyingChairs_train_val.txt', type=str, metavar='DIR', help='FlyingChairs_train_val.txt path')
parser.add_argument('--Sintel_train_val_path', default='./Sintel_train_val.txt', type=str, metavar='DIR', help='Sintel_train_val.txt path')

# training options
parser.add_argument('--num_epoches', default=1000, type=int, metavar='N', help='total training epoches (default: 1000)')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='training and testing batch size (default: 32)')
parser.add_argument('--test_interval', default=10, type=int, metavar='N', help='testing interval between training (default: 10)')
parser.add_argument('--checkpoint_interval', default=10, type=int, metavar='N', help='checkpoint interval between training (default: 10)')
parser.add_argument('--is_augment', default=True, type=str2bool, help='data augmentation option')
parser.add_argument('--use_pretrained', default=True, type=str2bool, help='use pretrained model option')
parser.add_argument('--cycle_train', default=False, type=str2bool, help='cycle train a model this will overwrite model_pretrained.pth')
parser.add_argument('--cudnn_benchmark', default=True, type=str2bool, help='cudnn benchmark option')

# optimization options
parser.add_argument('--optim_method', default='SGD', type=str, metavar='O', help='optimization method, select SGD or Adam')
parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='initial learning rate (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_scheduler', default='StepLR', type=str, metavar='LR Scheduler', help='learning rate scheduler, select StepLR or MultiStepLR (default: StepLR)')
parser.add_argument('--step_size', default=400, type=int, metavar='Step Size', help='StepLR step_size (default: 400)')
parser.add_argument('--milestones', default='400', metavar='Milestones', help='MultiStepLR milestones (default: [400])')
parser.add_argument('--gamma', default=0.1, type=float, metavar='Gamma', help='StepLR/MultiStepLR gamma (default: 0.1)')

# augmentation options
parser.add_argument('--angle', default=0.3, type=float, metavar='A', help='augment rotation angle in radians [-angle, +angle]')
parser.add_argument('--scale', default=15, type=int, metavar='S', help='augment scale in [1, 30/(31-scale)]')
parser.add_argument('--noise', default=0.1, type=float, metavar='N', help='augment noise added')
parser.add_argument('--brightness', default=0.4, type=float, metavar='B', help='augment colorjitter brightness')
parser.add_argument('--contrast', default=0.4, type=float, metavar='C', help='augment colorjitter contrast')
parser.add_argument('--saturation', default=0.4, type=float, metavar='S', help='augment colorjitter saturation')
parser.add_argument('--lighting', default=0.1, type=float, metavar='L', help='augment colorjitter lighting')



args=parser.parse_args()

if not os.path.exists(args.checkpoint_path):
    os.mkdir(args.checkpoint_path)

if not os.path.exists(args.result_path):
    os.mkdir(args.result_path)

args.parallel_gpu_ids = [int(item) for item in args.parallel_gpu_ids.split(',')]

args.milestones = [int(item) for item in args.milestones.split(',')]

if args.cycle_train:
    args.use_pretrained = True