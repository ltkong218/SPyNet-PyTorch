import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import FlyingChairsTrain, FlyingChairsTest, MPISintelTrain, MPISintelTest
from criterions import EPE
import copy
import os
import time
from arguments import args



flyingchairstest = FlyingChairsTest(args)
flyingchairstest_loader = DataLoader(flyingchairstest, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)




def test(model, args):
    print('num of testing data {}'.format(len(flyingchairstest)))

    batch_size = args.batch_size

    num_test_data = len(flyingchairstest)
    num_test_steps = int(np.ceil(num_test_data / batch_size))
    

    since = time.time()
    
    print('\n')
    # testing procedure
    print('\nTesting Level {}'.format(args.level))
    model.train(False)

    epoch_loss = 0.0
    counter = 0.0

    
    for i, data in enumerate(flyingchairstest_loader):
        imageFlowInputs, flowDiffOutput = data

        if args.is_parallel:
            imageFlowInputs = Variable(imageFlowInputs.cuda())
            flowDiffOutput = Variable(flowDiffOutput.cuda())
        else:
            imageFlowInputs = Variable(imageFlowInputs.cuda(args.net_gpu_id))
            flowDiffOutput = Variable(flowDiffOutput.cuda(args.net_gpu_id))

        FlowOutpus = model(imageFlowInputs)
        
        loss = EPE(flowDiffOutput, FlowOutpus)
        print('Testing Iteration {} / Total {},  Loss {:.6f}'.format(i+1, num_test_steps, loss.item()))
    
        epoch_loss += loss.item()
        counter += 1

    epoch_loss = epoch_loss / counter    
    print('\nTesting Loss {:.6f}'.format(epoch_loss))
            

    print('\n\n')
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed//3600, (time_elapsed%3600)//60, (time_elapsed%3600)%60))