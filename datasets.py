import os
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils import loadFlow, loadImage
from data import Augment, MakeData, Normalize, Resize


class FlyingChairsTrain(Dataset):
    def __init__(self, args):
        self.args = args
        self.augment = Augment(args)
        self.normalize = Normalize(args)
        self.makedata = MakeData(args)
        self.root_dir = args.flyingchairs_dataset_path
        self.img1_list_original = sorted(glob.glob(os.path.join(self.root_dir, '*_img1.ppm')))
        self.img2_list_original = sorted(glob.glob(os.path.join(self.root_dir, '*_img2.ppm')))
        self.flow_list_original = sorted(glob.glob(os.path.join(self.root_dir, '*_flow.flo')))
        
        self.img1_list = []
        self.img2_list = []
        self.flow_list = []

        self.train_test_split_list = np.loadtxt(args.FlyingChairs_train_val_path)

        # assert len(self.img1_list_original) == len(self.img2_list_original) and len(self.img2_list_original) == len(self.flow_list_original) \ 
        #     and len(self.flow_list_original) == len(self.train_test_split_list), 'Error: DataSet do not have the same size!'

        for i in range(len(self.flow_list_original)):
            if self.train_test_split_list[i] == 1:
            # if self.train_test_split_list[i] == 1 or 2:
                self.img1_list.append(self.img1_list_original[i])
                self.img2_list.append(self.img2_list_original[i])
                self.flow_list.append(self.flow_list_original[i])
            
        self.length = len(self.flow_list)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img1 = loadImage(self.img1_list[idx])
        img2 = loadImage(self.img2_list[idx])
        flow = loadFlow(self.flow_list[idx], 384, 512)

        if self.args.is_augment:
            img1_a, img2_a, flow_a = self.augment.augmentation(img1, img2, flow)
        else:
            img1_a, img2_a, flow_a = self.normalize.normalization(img1, img2, flow)
            
        imageFlowInputs, flowDiffOutput = self.makedata.makedata(img1_a, img2_a, flow_a)

        return imageFlowInputs.cpu(), flowDiffOutput.cpu()


class FlyingChairsTest(Dataset):
    def __init__(self, args):
        self.normalize = Normalize(args)
        self.makedata = MakeData(args)
        self.root_dir = args.flyingchairs_dataset_path
        self.img1_list_original = sorted(glob.glob(os.path.join(self.root_dir, '*_img1.ppm')))
        self.img2_list_original = sorted(glob.glob(os.path.join(self.root_dir, '*_img2.ppm')))
        self.flow_list_original = sorted(glob.glob(os.path.join(self.root_dir, '*_flow.flo')))
        
        self.img1_list = []
        self.img2_list = []
        self.flow_list = []

        self.train_test_split_list = np.loadtxt(args.FlyingChairs_train_val_path)

        # assert len(self.img1_list_original) == len(self.img2_list_original) and len(self.img2_list_original) == len(self.flow_list_original) \ 
        #     and len(self.flow_list_original) == len(self.train_test_split_list), 'Error: DataSet do not have the same size!'

        for i in range(len(self.flow_list_original)):
            if self.train_test_split_list[i] == 2:
                self.img1_list.append(self.img1_list_original[i])
                self.img2_list.append(self.img2_list_original[i])
                self.flow_list.append(self.flow_list_original[i])
            
        self.length = len(self.flow_list)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img1 = loadImage(self.img1_list[idx])
        img2 = loadImage(self.img2_list[idx])
        flow = loadFlow(self.flow_list[idx], 384, 512)

        img1_n, img2_n, flow_n = self.normalize.normalization(img1, img2, flow)

        imageFlowInputs, flowDiffOutput = self.makedata.makedata(img1_n, img2_n, flow_n)

        return imageFlowInputs.cpu(), flowDiffOutput.cpu()



class MPISintelTrain(Dataset):
    def __init__(self, args, dtype='clean'):
        self.args = args
        self.augment = Augment(args)
        self.normalize = Normalize(args)
        self.resize = Resize(args)
        self.makedata = MakeData(args)
        self.root_dir = args.mpisintel_dataset_path
        self.flow_root = os.path.join(self.root_dir, 'training', 'flow')
        self.image_root = os.path.join(self.root_dir, 'training', dtype)
        
        self.train_test_split_list = np.loadtxt(args.Sintel_train_val_path)
        self.file_list = sorted(glob.glob(os.path.join(self.flow_root, '*/*.flo')))
        
        self.flow_list = []
        self.image_list = []
        
        for i, file in enumerate(self.file_list):
            if self.train_test_split_list[i] == 1:
                fbase = file[len(self.flow_root)+1:]
                fprefix = fbase[:-8]
                fnum = int(fbase[-8:-4])
                
                img1 = os.path.join(self.image_root, fprefix + '%04d' % (fnum+0) + '.png')
                img2 = os.path.join(self.image_root, fprefix + '%04d' % (fnum+1) + '.png')
                            
                if not os.path.isfile(img1) or not os.path.isfile(img2) or not os.path.isfile(file):
                    continue
                    
                self.image_list += [[img1, img2]]
                self.flow_list += [file]
            
        self.size = len(self.image_list)
        
        assert len(self.image_list) == len(self.flow_list)

    def __len__(self):
        return self.size
        
    def __getitem__(self, index):
        img1 = loadImage(self.image_list[index][0])
        img2 = loadImage(self.image_list[index][1])
        flow = loadFlow(self.flow_list[index], 436, 1024)

        if img1.size(1) % 32 != 0 or img1.size(2) % 32 != 0:
            img1, img2, flow = self.resize.resize(img1, img2, flow)
        
        if self.args.is_augment:
            img1_a, img2_a, flow_a = self.augment.augmentation(img1, img2, flow)
        else:
            img1_a, img2_a, flow_a = self.normalize.normalization(img1, img2, flow)
        
        imageFlowInputs, flowDiffOutput = self.makedata.makedata(img1_a, img2_a, flow_a)

        return imageFlowInputs.cpu(), flowDiffOutput.cpu()



class MPISintelTest(Dataset):
    def __init__(self, args, dtype='clean'):
        self.args = args
        self.normalize = Normalize(args)
        self.resize = Resize(args)
        self.makedata = MakeData(args)
        self.root_dir = args.mpisintel_dataset_path
        self.flow_root = os.path.join(self.root_dir, 'training', 'flow')
        self.image_root = os.path.join(self.root_dir, 'training', dtype)
        
        self.train_test_split_list = np.loadtxt(args.Sintel_train_val_path)
        self.file_list = sorted(glob.glob(os.path.join(self.flow_root, '*/*.flo')))
        
        self.flow_list = []
        self.image_list = []
        
        for i, file in enumerate(self.file_list):
            if self.train_test_split_list[i] == 2:
                fbase = file[len(self.flow_root)+1:]
                fprefix = fbase[:-8]
                fnum = int(fbase[-8:-4])
                
                img1 = os.path.join(self.image_root, fprefix + '%04d' % (fnum+0) + '.png')
                img2 = os.path.join(self.image_root, fprefix + '%04d' % (fnum+1) + '.png')
                            
                if not os.path.isfile(img1) or not os.path.isfile(img2) or not os.path.isfile(file):
                    continue
                    
                self.image_list += [[img1, img2]]
                self.flow_list += [file]
            
        self.size = len(self.image_list)
        
        assert len(self.image_list) == len(self.flow_list)

    def __len__(self):
        return self.size
        
    def __getitem__(self, index):
        img1 = loadImage(self.image_list[index][0])
        img2 = loadImage(self.image_list[index][1])
        flow = loadFlow(self.flow_list[index], 436, 1024)
        
        if img1.size(1) % 32 != 0 or img1.size(2) % 32 != 0:
            img1, img2, flow = self.resize.resize(img1, img2, flow)
        
        img1_n, img2_n, flow_n = self.normalize.normalization(img1, img2, flow)

        imageFlowInputs, flowDiffOutput = self.makedata.makedata(img1_n, img2_n, flow_n)

        return imageFlowInputs.cpu(), flowDiffOutput.cpu()