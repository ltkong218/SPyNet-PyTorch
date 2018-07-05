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
from visdom import Visdom


flyingchairstrain = FlyingChairsTrain(args)
flyingchairstrain_loader = DataLoader(flyingchairstrain, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

flyingchairstest = FlyingChairsTest(args)
flyingchairstest_loader = DataLoader(flyingchairstest, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

mpisinteltrain = MPISintelTrain(args)
mpisinteltrain_loader = DataLoader(mpisinteltrain, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

mpisinteltest = MPISintelTest(args)
mpisinteltest_loader = DataLoader(mpisinteltest, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)




def train(model, optimizer, scheduler, args):
    print('num of training data {}'.format(len(flyingchairstrain)))
    # print('num of training data {}'.format(len(mpisinteltrain)))
    print('num of testing data {}'.format(len(flyingchairstest)))
    torch.manual_seed(args.manual_seed)

    batch_size = args.batch_size

    num_train_data = len(flyingchairstrain)
    # num_train_data = len(mpisinteltrain)
    num_train_steps = int(np.ceil(num_train_data / batch_size))
    num_test_data = len(flyingchairstest)
    num_test_steps = int(np.ceil(num_test_data / batch_size))

    num_epoches = args.num_epoches
    
    best_model_wts = copy.deepcopy(model.state_dict())
    least_testing_loss = 100.0

    train_epoch_loss_list = []
    test_epoch_loss_list = []

    # visdom
    if args.use_visdom:
        viz=Visdom()

        win = viz.line(X=np.zeros((1)), Y=np.zeros((1)), opts=dict(xlabel='epoch', ylabel='training/testing loss', title='Training & Testing Loss', width=1500, height=400)) 

    
    since = time.time()
    
    print('\n')
    
    for epoch in range(num_epoches):
        print('Training Epoch {} / {}'.format(epoch+1, num_epoches))
        print('-' * 20)

        # training procedure
        print('\nTraining Level {}'.format(args.level))
        model.train(True)

        scheduler.step()
        
        epoch_loss = 0.0
        counter = 0.0
        
        for i, data in enumerate(flyingchairstrain_loader):
        # for i, data in enumerate(mpisinteltrain_loader):
            imageFlowInputs, flowDiffOutput = data
            
            if args.is_parallel:
                imageFlowInputs = Variable(imageFlowInputs.cuda())
                flowDiffOutput = Variable(flowDiffOutput.cuda())
            else:
                imageFlowInputs = Variable(imageFlowInputs.cuda(args.net_gpu_id))
                flowDiffOutput = Variable(flowDiffOutput.cuda(args.net_gpu_id))
            
            optimizer.zero_grad()

            FlowOutputs = model(imageFlowInputs)
            
            loss = EPE(flowDiffOutput, FlowOutputs)
            print('Training Iteration {} / Total {},  Loss {:.6f}'.format(i+1, num_train_steps, loss.item()))
            
            epoch_loss += loss.item()
            counter += 1
            
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / counter    
        train_epoch_loss_list.append(epoch_loss)
        print('\nTraining Epoch {} / Total {},  Loss {:.6f}'.format(epoch+1, num_epoches, epoch_loss))

        if args.use_visdom:
            viz.line(X=np.ones((1))*(epoch+1), Y=np.array(epoch_loss).reshape(1), win=win, name='train loss', update='append')
        
        # save training epoch loss to a txt file
        file_train = open(os.path.join(args.result_path, 'training_epoch_loss.txt'), 'a')
        file_train.write('Training Epoch {} / Total {},  Loss {:.6f}'.format(epoch+1, num_epoches, epoch_loss) + '\n')
        file_train.close()

        if (epoch+1) % args.test_interval == 0:
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

                FlowOutputs = model(imageFlowInputs)
                
                loss = EPE(flowDiffOutput, FlowOutputs)
                print('Testing Iteration {} / Total {},  Loss {:.6f}'.format(i+1, num_test_steps, loss.item()))
            
                epoch_loss += loss.item()
                counter += 1

            epoch_loss = epoch_loss / counter    
            test_epoch_loss_list.append(epoch_loss)
            print('\nTesting Loss {:.6f}'.format(epoch_loss))

            if args.use_visdom:
                viz.line(X=np.ones((1))*(epoch+1), Y=np.array(epoch_loss).reshape(1), win=win, name='test loss', update='append')

            # save testing epoch loss to a txt file
            file_test = open(os.path.join(args.result_path, 'testing_epoch_loss.txt'), 'a')
            file_test.write('Testing Epoch {} / Total {},  Loss {:.6f}'.format(epoch+1, num_epoches, epoch_loss) + '\n')
            file_test.close()
            
            # update best model on test dataset
            if epoch_loss < least_testing_loss:
                least_testing_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


        # save checkpoint every interval epoches
        if (epoch+1) % args.checkpoint_interval == 0:
            checkpoint_name = 'model' + str(args.level) + '_epoch_' + str(epoch+1) + '.pth'
            checkpoint_path = os.path.join(args.checkpoint_path, checkpoint_name)
            torch.save(model.state_dict(), checkpoint_path)
        
        print('\n\n')

    print('\n\n')
    time_elapsed = time.time() - since
    print('Training and Testing complete in {:.0f}h {:.0f}m {:.0f}s\n'.format(time_elapsed//3600, (time_elapsed%3600)//60, (time_elapsed%3600)%60))
    print('Best Testing Loss: {:.6f}\n\n'.format(least_testing_loss))

    
    # save best model parameters
    best_model_path = 'model' + str(args.level) + '_best.pth'
    best_model_path = os.path.join(args.result_path, best_model_path)
    torch.save(best_model_wts, best_model_path)

    return model