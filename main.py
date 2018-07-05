import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
from modules import Net, loadTorchModel
from arguments import args
import torch.backends.cudnn as cudnn



if __name__ == '__main__':
    # multiprocessing with cuda
    mp.set_start_method('spawn')

    # cudnn benchmark
    cudnn.benchmark = args.cudnn_benchmark
    
    from train import train
    from test import test

    torch.manual_seed(args.manual_seed)

    # use data parallel
    if args.is_parallel:
        torch.cuda.set_device(args.parallel_gpu_ids[0])

        if args.use_pretrained:
            model = Net()
            if args.use_pytorch_model:
                model.load_state_dict(torch.load('./model_pretrained.pth', map_location=lambda storage, loc: storage))
            else:
                model = loadTorchModel('./model_pretrained.t7')
        else:
            model = Net()
        
        model = torch.nn.DataParallel(model, device_ids=args.parallel_gpu_ids).cuda()

    # not use data parallel
    else:
        model = Net().cuda(args.net_gpu_id)

        if args.use_pretrained:
            model = Net()
            if args.use_pytorch_model:
                model.load_state_dict(torch.load('./model_pretrained.pth', map_location=lambda storage, loc: storage))
            else:
                model = loadTorchModel('./model_pretrained.t7')
        else:
            model = Net()

        model = model.cuda(args.net_gpu_id)



    if args.mode == 'train':
        # optimization method
        if args.optim_method == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optim_method == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # learning rate scheduler
        if args.lr_scheduler == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        if args.lr_scheduler == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        
        # show configuration in terminal
        print('Arguments')
        print('level={}'.format(args.level))
        print('num_epoches={}'.format(args.num_epoches))
        print('use_pretrained={}'.format(args.use_pretrained))
        print('cycle_train={}'.format(args.cycle_train))
        print('is_parallel={}'.format(args.is_parallel))
        if args.is_parallel:
            print('parallel_gpu_ids={}'.format(args.parallel_gpu_ids))
        print('use_visdom={}'.format(args.use_visdom))

        print('optim_method={}'.format(args.optim_method))
        if args.optim_method == 'SGD':
            print('initial_lr={}'.format(args.lr))
            print('momentum={}'.format(args.momentum))
            print('weight_decay={}'.format(args.weight_decay))
        if args.optim_method == 'Adam':
            print('initial_lr={}'.format(args.lr))

        print('lr_scheduler={}'.format(args.lr_scheduler))
        if args.lr_scheduler == 'StepLR':
            print('step_size={}'.format(args.step_size))
            print('gamma={}'.format(args.gamma))
        if args.lr_scheduler == 'MultiStepLR':
            print('milestones={}'.format(args.milestones))
            print('gamma={}'.format(args.gamma))

        print('is_augment={}'.format(args.is_augment))
        if args.is_augment:
            print('angle={}'.format(args.angle))
            print('scale={}'.format(args.scale))
            print('noise={}'.format(args.noise))
            print('brightness={}'.format(args.brightness))
            print('contrast={}'.format(args.contrast))
            print('saturation={}'.format(args.saturation))
            print('lighting={}'.format(args.lighting))
        print('\n\n\n')

        # write configuration in file
        file_config = open(os.path.join(args.result_path, 'configuration.txt'), 'a')

        file_config.write('Arguments\n')
        file_config.write('level={}\n'.format(args.level))
        file_config.write('num_epoches={}\n'.format(args.num_epoches))
        file_config.write('use_pretrained={}\n'.format(args.use_pretrained))
        file_config.write('cycle_train={}\n'.format(args.cycle_train))
        file_config.write('is_parallel={}\n'.format(args.is_parallel))
        if args.is_parallel:
            file_config.write('parallel_gpu_ids={}\n'.format(args.parallel_gpu_ids))
        file_config.write('use_visdom={}\n'.format(args.use_visdom))

        file_config.write('optim_method={}\n'.format(args.optim_method))
        if args.optim_method == 'SGD':
            file_config.write('initial_lr={}\n'.format(args.lr))
            file_config.write('momentum={}\n'.format(args.momentum))
            file_config.write('weight_decay={}\n'.format(args.weight_decay))
        if args.optim_method == 'Adam':
            file_config.write('initial_lr={}\n'.format(args.lr))

        file_config.write('lr_scheduler={}\n'.format(args.lr_scheduler))
        if args.lr_scheduler == 'StepLR':
            file_config.write('step_size={}\n'.format(args.step_size))
            file_config.write('gamma={}\n'.format(args.gamma))
        if args.lr_scheduler == 'MultiStepLR':
            file_config.write('milestones={}\n'.format(args.milestones))
            file_config.write('gamma={}\n'.format(args.gamma))

        file_config.write('is_augment={}\n'.format(args.is_augment))
        if args.is_augment:
            file_config.write('angle={}\n'.format(args.angle))
            file_config.write('scale={}\n'.format(args.scale))
            file_config.write('noise={}\n'.format(args.noise))
            file_config.write('brightness={}\n'.format(args.brightness))
            file_config.write('contrast={}\n'.format(args.contrast))
            file_config.write('saturation={}\n'.format(args.saturation))
            file_config.write('lighting={}\n'.format(args.lighting))
        file_config.write('\n\n\n')

        file_config.close()


        model = train(model, optimizer, scheduler, args)


        model_final_path = 'model' + str(args.level) + '_final' + '.pth'
        model_final_path = os.path.join(args.result_path ,model_final_path)
        torch.save(model.state_dict(), model_final_path)

        if args.cycle_train:
            torch.save(model.state_dict(), './model_pretrained.pth')

    
    if args.mode == 'test':
        test(model, args)