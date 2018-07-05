import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import argparse
from flowextension import computeImg


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.Resize((384, 512)), transforms.ToTensor()])


def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentError('Boolean value expected')


parser = argparse.ArgumentParser(description='Test SPyNet')

parser.add_argument('--model_path', type=str, default='./models/myClean', help='model path')
parser.add_argument('--use_pytorch_model', type=str2bool, default=True, help='use pytorch model or use torch model')
parser.add_argument('--img1_path', type=str, default='./data/img1.jpg', help='img1 path')
parser.add_argument('--img2_path', type=str, default='./data/img2.jpg', help='img2 path')
parser.add_argument('--flo_path', type=str, default='./eval_result', help='flo file output path')
parser.add_argument('--flow_path', type=str, default='./eval_result', help='flow img output path')

args=parser.parse_args()

if not os.path.exists(args.flo_path):
    os.mkdir(args.flo_path)

if not os.path.exists(args.flow_path):
    os.mkdir(args.flow_path)

    


class SPyNet(nn.Module):
    
    def __init__(self, model_path, use_pytorch_model=True):
        super(SPyNet, self).__init__()
        self.model_path = model_path
        self.use_pytorch_model = use_pytorch_model
        
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2).eval()
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2).eval()
        self.down4 = nn.AvgPool2d(kernel_size=2, stride=2).eval()
        self.down5 = nn.AvgPool2d(kernel_size=2, stride=2).eval()
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).eval()
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).eval()
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).eval()
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).eval()
        
        if self.use_pytorch_model:
            self.modelL1 = loadPyTorchModel(os.path.join(self.model_path, 'model1.pth')).eval()
            self.modelL2 = loadPyTorchModel(os.path.join(self.model_path, 'model2.pth')).eval()
            self.modelL3 = loadPyTorchModel(os.path.join(self.model_path, 'model3.pth')).eval()
            self.modelL4 = loadPyTorchModel(os.path.join(self.model_path, 'model4.pth')).eval()
            self.modelL5 = loadPyTorchModel(os.path.join(self.model_path, 'model5.pth')).eval()
        else:
            self.modelL1 = loadTorchModel(os.path.join(self.model_path, 'model1.t7')).eval()
            self.modelL2 = loadTorchModel(os.path.join(self.model_path, 'model2.t7')).eval()
            self.modelL3 = loadTorchModel(os.path.join(self.model_path, 'model3.t7')).eval()
            self.modelL4 = loadTorchModel(os.path.join(self.model_path, 'model4.t7')).eval()
            self.modelL5 = loadTorchModel(os.path.join(self.model_path, 'model5.t7')).eval()


    def createWarpModel(self, img, flow):
        # convert flow from 12HW to 1HW2
        flow = flow.transpose(1, 2).transpose(2, 3)
        
        batch_size = img.size(0)
        height = img.size(2)
        width = img.size(3)
        # scale factors 
        K_x = 2.0 / (width - 1)
        K_y = 2.0 / (height - 1)

        # generate base flow of size 1HW2
        x = torch.linspace(-1.0, 1.0, width).cuda().view(1, 1, width, 1).\
            expand(batch_size, height, width, 1)
        y = torch.linspace(-1.0, 1.0, height).cuda().view(1, height, 1, 1).\
            expand(batch_size, height, width, 1)

        base_flow = torch.cat([x, y], 3)

        flow_n = flow.clone()
        flow_n[:,:,:,0] = flow[:,:,:,0] * K_x
        flow_n[:,:,:,1] = flow[:,:,:,1] * K_y

        flow_added = base_flow + flow_n.data
        
        # img size: BCHW, flow size: 1HW2
        # the first dimension is horizontal x, the second dimension is vertical y
        imgout = F.grid_sample(img, flow_added, mode='bilinear', padding_mode='border')

        return imgout


    def computeInitFlowL1(self, imagesL1):
        h = imagesL1.size()[2]
        w = imagesL1.size()[3]
        batchSize = imagesL1.size()[0]
        
        _flowappend = Variable(torch.zeros(batchSize, 2, h, w).type(torch.cuda.FloatTensor))
        
        images_in = torch.cat((imagesL1, _flowappend), 1)
        flow_est = self.modelL1.forward(images_in)
        return flow_est
    
    def computeInitFlowL2(self, imagesL2):
        imagesL1 = self.down2.forward(imagesL2)
        _flowappend = self.up2.forward(self.computeInitFlowL1(imagesL1)) * 2
        
        _img2 = imagesL2[:, 3:6, :, :]
        imagesL2_c = imagesL2.clone()

        imagesL2_c[:, 3:6, :, :] = (self.createWarpModel(_img2, _flowappend))
        
        images_in = torch.cat((imagesL2_c, _flowappend), 1)
        flow_est = self.modelL2.forward(images_in)
        return (flow_est + _flowappend)
    
    def computeInitFlowL3(self, imagesL3):
        imagesL2 = self.down3.forward(imagesL3.clone())
        _flowappend = self.up3.forward(self.computeInitFlowL2(imagesL2)) * 2
        
        _img2 = imagesL3[:, 3:6, :, :]
        imagesL3_c = imagesL3.clone()
        
        imagesL3_c[:, 3:6, :, :] = (self.createWarpModel(_img2, _flowappend))
        
        images_in = torch.cat((imagesL3_c, _flowappend), 1)
        flow_est = self.modelL3.forward(images_in)
        return (flow_est + _flowappend)
    
    def computeInitFlowL4(self, imagesL4):
        imagesL3 = self.down4.forward(imagesL4.clone())
        _flowappend = self.up4.forward(self.computeInitFlowL3(imagesL3)) * 2
        
        _img2 = imagesL4[:, 3:6, :, :]
        imagesL4_c = imagesL4.clone()
        
        imagesL4_c[:, 3:6, :, :] = (self.createWarpModel(_img2, _flowappend))
        
        images_in = torch.cat((imagesL4_c, _flowappend), 1)
        flow_est = self.modelL4.forward(images_in)
        return (flow_est + _flowappend)
    
    def computeInitFlowL5(self, imagesL5):
        imagesL4 = self.down5.forward(imagesL5.clone())
        _flowappend = self.up5.forward(self.computeInitFlowL4(imagesL4)) * 2
        
        _img2 = imagesL5[:, 3:6, :, :]
        imagesL5_c = imagesL5.clone()
        
        imagesL5_c[:, 3:6, :, :] = (self.createWarpModel(_img2, _flowappend))
        
        images_in = torch.cat((imagesL5_c, _flowappend), 1)
        flow_est = self.modelL5.forward(images_in)
        return (flow_est + _flowappend)


    def forward(self, imgs):
        flow_est = self.computeInitFlowL5(imgs)
        return flow_est



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('conv3', nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('conv4', nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3))
        self.model.add_module('relu4', nn.ReLU())
        self.model.add_module('conv5', nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3))
        
        
    def forward(self, input):
        output = self.model(input)
        return output



def loadTorchModel(torch_model_path):
    net_pytorch = Net()
    model_pytorch = net_pytorch.model
    model_torch = load_lua(torch_model_path)

    assert len(model_torch) == len(model_pytorch), 'Error: Torch Module and PyTorch Module have different Lengths!'

    for i in range(len(model_torch)):
        if i % 2 == 0:
            model_pytorch[i].weight = torch.nn.Parameter(model_torch.modules[i].weight)
            model_pytorch[i].bias = torch.nn.Parameter(model_torch.modules[i].bias)

    return net_pytorch


def loadPyTorchModel(pytorch_model_path):
    net_pytorch = Net()
    net_pytorch.load_state_dict(torch.load(pytorch_model_path, map_location=lambda storage, loc: storage))
    return net_pytorch



TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=int(2*w*h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()



if __name__ == '__main__':
    
    spynet = SPyNet(args.model_path, args.use_pytorch_model).cuda()

    img1 = Image.open(args.img1_path)
    img2 = Image.open(args.img2_path)

    img1 = transform(img1)
    img2 = transform(img2)

    img1_np = ((torch.squeeze(img1)).numpy()).transpose(1, 2, 0)
    img2_np = ((torch.squeeze(img2)).numpy()).transpose(1, 2, 0)

    img1_n = torch.from_numpy((img1_np - mean) / std).permute(2, 0, 1).type(torch.FloatTensor)
    img2_n = torch.from_numpy((img2_np - mean) / std).permute(2, 0, 1).type(torch.FloatTensor)

    imgs = torch.cat((img1_n, img2_n), 0)
    imgs = imgs.unsqueeze(0)
    imgs = Variable(imgs, requires_grad=False).cuda()

    flow_est = spynet(imgs)
    
    flow_show = flow_est.data.squeeze().cpu().numpy().transpose(1, 2, 0)
    # flow_show = readFlow('./output.flo')
    img = computeImg(flow_show)

    flow2write = flow_est.cpu().data.squeeze(0).permute(1, 2, 0).numpy()
    writeFlow(os.path.join(args.flo_path, 'output.flo'), flow2write)

    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(args.flow_path, 'flow.png'))


    fig, axes = plt.subplots(nrows=1, ncols=3, subplot_kw={'xticks':[], 'yticks':[]}, figsize=(18, 5))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes[0].imshow(img1_np)
    axes[0].set_title('img1')
    axes[1].imshow(img2_np)
    axes[1].set_title('img2')
    axes[2].imshow(img)
    axes[2].set_title('SPyNet flow')
    plt.show()