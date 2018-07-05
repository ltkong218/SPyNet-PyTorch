import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from modules import Net



class Resize():
    def __init__(self, args):
        self.args = args
        self.gpu_id = args.aug_make_data_gpu_id

    def bchw_resize(self, img, height, width):
        x = torch.linspace(-1.0, 1.0, width).cuda(self.gpu_id).view(1, 1, width, 1).\
            expand(1, height, width, 1)
        y = torch.linspace(-1.0, 1.0, height).cuda(self.gpu_id).view(1, height, 1, 1).\
            expand(1, height, width, 1)

        grid = torch.cat([x, y], 3)

        img_rotated_resized = F.grid_sample(img, grid, mode='bilinear', padding_mode='border').data
        return img_rotated_resized

    def resize(self, img1, img2, flow):
        img1 = img1.cuda(self.gpu_id)
        img2 = img2.cuda(self.gpu_id)
        flow = flow.cuda(self.gpu_id)

        height = img1.size(1)
        width = img1.size(2)

        if width % 32 == 0:
            width = width
        else:
            width = width + 32 - width % 32

        if height % 32 == 0:
            height = height
        else:
            height = height + 32 - height % 32

        sc_x = float(width) / img1.size(2)
        sc_y = float(height) / img1.size(1)

        
        data = torch.cat((img1, img2, flow), 0)

        data = self.bchw_resize(data.unsqueeze(0), height, width).squeeze()

        data[6,:,:] = data[6,:,:] * sc_x
        data[7,:,:] = data[7,:,:] * sc_y
        
        img1_r, img2_2, flow_r = data[:3,:,:], data[3:6,:,:], data[6:,:,:]
        return img1_r, img2_2, flow_r



class Augment():
    def __init__(self, args):
        self.gpu_id = args.aug_make_data_gpu_id
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda(self.gpu_id)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda(self.gpu_id)
        self.eigval = torch.FloatTensor([0.2175, 0.0188, 0.0045]).cuda(self.gpu_id)
        self.eigvec = torch.FloatTensor([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]).cuda(self.gpu_id)
        self.angle = args.angle
        self.scale = args.scale
        self.noise = args.noise
        self.brightness = args.brightness
        self.contrast = args.contrast
        self.saturation = args.saturation
        self.lighting = args.lighting


    def blend(self, img1, img2, alpha):
        return alpha * img1 + (1 - alpha) * img2
    
    def grayscale(self, img):
        assert img.size(0) == 3
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        gray.unsqueeze_(0)
        gray = torch.cat((gray, gray, gray), 0)
        return gray

    def Lighting(self, img, alphastd):
        if alphastd == 0:
            return img

        alpha = torch.Tensor(3).normal_(mean=0, std=alphastd).cuda(self.gpu_id)

        rgb = torch.mul(self.eigvec, alpha.view(1, 3).expand(3, 3))
        rgb = torch.mul(rgb, self.eigval.view(1, 3).expand(3, 3)).sum(1)

        for i in range(3):
            img[i] += rgb[i]
            img[i+3] += rgb[i]

        return img

    def Saturation(self, img, var):
        img_gray = img.clone()
        img_gray[:3] = self.grayscale(img[:3])
        img_gray[3:6] = self.grayscale(img[3:6])

        alpha = ((torch.Tensor(1).uniform_() * 2 - 1) * var + 1.0).cuda(self.gpu_id)

        img = self.blend(img, img_gray, alpha)
        return img

    def Brightness(self, img, var):
        img_zero = img.clone().zero_()
        
        alpha = ((torch.Tensor(1).uniform_() * 2 - 1) * var + 1.0).cuda(self.gpu_id)

        img = self.blend(img, img_zero, alpha)
        return img

    def Contrast(self, img, var):
        img_mean = img.clone()
        img_mean[:3] = self.grayscale(img[:3])
        img_mean[3:6] = self.grayscale(img[3:6])

        img_mean[:3].fill_(img_mean[0].mean())
        img_mean[3:6].fill_(img_mean[3].mean())

        alpha = ((torch.Tensor(1).uniform_() * 2 - 1) * var + 1.0).cuda(self.gpu_id)
        
        img = self.blend(img, img_mean, alpha)
        return img
    
    def ColorJitter(self, img, brightness, contrast, saturation, lighting):
        img = self.Brightness(img, brightness)
        img = self.Contrast(img, contrast)
        img = self.Saturation(img, saturation)
        img = self.Lighting(img, lighting)
        return img

    
    def bchw_rotate_resize(self, img, angle, height, width):
        x = torch.linspace(-1.0, 1.0, width).cuda(self.gpu_id).view(1, 1, width, 1).\
            expand(1, height, width, 1)
        y = torch.linspace(-1.0, 1.0, height).cuda(self.gpu_id).view(1, height, 1, 1).\
            expand(1, height, width, 1)

        x_ = torch.cos(angle) * x - torch.sin(angle) * y
        y_ = torch.sin(angle) * x + torch.cos(angle) * y

        grid = torch.cat([x_, y_], 3)

        img_rotated_resized = F.grid_sample(img, grid, mode='bilinear', padding_mode='border').data
        return img_rotated_resized
    

    def augmentation(self, img1, img2, flow):
        img1 = img1.cuda(self.gpu_id)
        img2 = img2.cuda(self.gpu_id)
        flow = flow.cuda(self.gpu_id)

        height = img1.size(1)
        width = img1.size(2)

        imgs = torch.cat((img1, img2), 0)

        # rotation [-angle, +angle] in radians
        angle = torch.Tensor(1).uniform_().cuda(self.gpu_id) * 2 * self.angle - self.angle
        # random scale
        sc = np.floor(np.random.uniform(1e-2, self.scale))
        sc = 30 / (sc + (30 - self.scale + 1))
        scale_height = int(sc * height)
        scale_width = int(sc * width)

        imgs_s = self.bchw_rotate_resize(imgs.unsqueeze(0), angle, scale_height, scale_width).squeeze()
        flow_s = self.bchw_rotate_resize(flow.unsqueeze(0), angle, scale_height, scale_width).squeeze() * sc

        # add random noise
        imgs_n = imgs_s + torch.rand(imgs_s.size()).cuda(self.gpu_id) * self.noise
        # imgs_n.clamp_(0, 1)

        # do random crop
        hI = np.random.randint(0, scale_height - height + 1)
        wI = np.random.randint(0, scale_width - width + 1)

        imgs_c = imgs_n[:, hI:hI+height, wI:wI+width]
        flow_c = flow_s[:, hI:hI+height, wI:wI+width]

        # color jitter
        imgs_j = self.ColorJitter(imgs_c, brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, lighting=self.lighting)

        img1_j = imgs_j[:3]
        img2_j = imgs_j[3:6]

        # normalize
        img1_n = ((img1_j.transpose(0, 1).transpose(1, 2) - self.mean) / self.std)\
                           .transpose(1, 2).transpose(0, 1)

        img2_n = ((img2_j.transpose(0, 1).transpose(1, 2) - self.mean) / self.std)\
                           .transpose(1, 2).transpose(0, 1)


        return img1_n, img2_n, flow_c


class Normalize():
    def __init__(self, args):
        self.gpu_id = args.aug_make_data_gpu_id
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).cuda(self.gpu_id)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).cuda(self.gpu_id)

    def normalization(self, img1, img2, flow):
        img1 = img1.cuda(self.gpu_id)
        img2 = img2.cuda(self.gpu_id)
        flow = flow.cuda(self.gpu_id)

        # normalize
        img1_n = ((img1.transpose(0, 1).transpose(1, 2) - self.mean) / self.std)\
                           .transpose(1, 2).transpose(0, 1)

        img2_n = ((img2.transpose(0, 1).transpose(1, 2) - self.mean) / self.std)\
                           .transpose(1, 2).transpose(0, 1)
        
        return img1_n, img2_n, flow



class MakeData():
    def __init__(self, args):
        self.gpu_id = args.aug_make_data_gpu_id
        self.level = args.level
        # self.FineHeight = args.FineHeight
        # self.FineWidth = args.FineWidth
        self.model_path = args.model_path
        
        self.downs2 = nn.AvgPool2d(kernel_size=2, stride=2).cuda(self.gpu_id).eval()
        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()
        self.downs4 = nn.AvgPool2d(kernel_size=4, stride=4).cuda(self.gpu_id).eval()
        self.ups4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()
        self.downs8 = nn.AvgPool2d(kernel_size=8, stride=8).cuda(self.gpu_id).eval()
        self.ups8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()
        self.downs16 = nn.AvgPool2d(kernel_size=16, stride=16).cuda(self.gpu_id).eval()
        self.ups16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()
        
        if self.level > 1:
            if args.use_pytorch_model:
                self.modelL1 = self.loadPyTorchModel(os.path.join(self.model_path, 'model1.pth')).cuda(self.gpu_id).eval()
            else:
                self.modelL1 = self.loadTorchModel(os.path.join(self.model_path, 'model1.t7')).cuda(self.gpu_id).eval()
            self.down2 = nn.AvgPool2d(kernel_size=2, stride=2).cuda(self.gpu_id).eval()
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()

        if self.level > 2:
            if args.use_pytorch_model:
                self.modelL2 = self.loadPyTorchModel(os.path.join(self.model_path, 'model2.pth')).cuda(self.gpu_id).eval()
            else:
                self.modelL2 = self.loadTorchModel(os.path.join(self.model_path, 'model2.t7')).cuda(self.gpu_id).eval()
            self.down3 = nn.AvgPool2d(kernel_size=2, stride=2).cuda(self.gpu_id).eval()
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()

        if self.level > 3:
            if args.use_pytorch_model:
                self.modelL3 = self.loadPyTorchModel(os.path.join(self.model_path, 'model3.pth')).cuda(self.gpu_id).eval()
            else:
                self.modelL3 = self.loadTorchModel(os.path.join(self.model_path, 'model3.t7')).cuda(self.gpu_id).eval()
            self.down4 = nn.AvgPool2d(kernel_size=2, stride=2).cuda(self.gpu_id).eval()
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()

        if self.level > 4:
            if args.use_pytorch_model:
                self.modelL4 = self.loadPyTorchModel(os.path.join(self.model_path, 'model4.pth')).cuda(self.gpu_id).eval()
            else:
                self.modelL4 = self.loadTorchModel(os.path.join(self.model_path, 'model4.t7')).cuda(self.gpu_id).eval()
            self.down5 = nn.AvgPool2d(kernel_size=2, stride=2).cuda(self.gpu_id).eval()
            self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()

        if self.level > 5:
            if args.use_pytorch_model:
                self.modelL5 = self.loadPyTorchModel(os.path.join(self.model_path, 'model5.pth')).cuda(self.gpu_id).eval()
            else:
                self.modelL5 = self.loadTorchModel(os.path.join(self.model_path, 'model5.t7')).cuda(self.gpu_id).eval()
            self.down6 = nn.AvgPool2d(kernel_size=2, stride=2).cuda(self.gpu_id).eval()
            self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False).cuda(self.gpu_id).eval()
            
            
    def image_warp(self, img, flow):
        # convert flow from B2HW to BHW2
        flow = flow.permute(0, 2, 3, 1)
        
        batch_size = img.size(0)
        height = img.size(2)
        width = img.size(3)

        # scale factors 
        K_x = 2.0 / (width - 1)
        K_y = 2.0 / (height - 1)

        # generate base flow of size BHW2
        x = torch.linspace(-1.0, 1.0, width).cuda(self.gpu_id).view(1, 1, width, 1).\
            expand(batch_size, height, width, 1)
        y = torch.linspace(-1.0, 1.0, height).cuda(self.gpu_id).view(1, height, 1, 1).\
            expand(batch_size, height, width, 1)

        base_flow = torch.cat([x, y], 3)

        flow_n = flow.clone()
        flow_n[:,:,:,0] = flow[:,:,:,0] * K_x
        flow_n[:,:,:,1] = flow[:,:,:,1] * K_y

        flow_added = base_flow + flow_n.data
        
        # img size: BCHW, flow size: BHW2
        # the first dimension is horizontal x, the second dimension is vertical y
        imgout = F.grid_sample(img, flow_added, mode='bilinear', padding_mode='border')
        return imgout
    
    def computeInitFlowL1(self, imagesL1):
        h = imagesL1.size()[2]
        w = imagesL1.size()[3]
        batchSize = imagesL1.size()[0]
        
        _flowappend = Variable(torch.zeros(batchSize, 2, h, w).type(torch.FloatTensor)).cuda(self.gpu_id)
        
        images_in = torch.cat((imagesL1, _flowappend), 1)
        flow_est = self.modelL1.forward(images_in)
        return flow_est
    
    def computeInitFlowL2(self, imagesL2):
        imagesL1 = self.down2.forward(imagesL2)
        _flowappend = self.up2.forward(self.computeInitFlowL1(imagesL1)) * 2

        _img2 = imagesL2[:, 3:6, :, :]
        imagesL2_c = imagesL2.clone()

        imagesL2_c[:, 3:6, :, :] = self.image_warp(_img2, _flowappend)
        
        images_in = torch.cat((imagesL2_c, _flowappend), 1)
        flow_est = self.modelL2.forward(images_in)
        return (flow_est + _flowappend)
    
    def computeInitFlowL3(self, imagesL3):
        imagesL2 = self.down3.forward(imagesL3.clone())
        _flowappend = self.up3.forward(self.computeInitFlowL2(imagesL2)) * 2
        
        _img2 = imagesL3[:, 3:6, :, :]
        imagesL3_c = imagesL3.clone()
        
        imagesL3_c[:, 3:6, :, :] = self.image_warp(_img2, _flowappend)
        
        images_in = torch.cat((imagesL3_c, _flowappend), 1)
        flow_est = self.modelL3.forward(images_in)
        return (flow_est + _flowappend)
    
    def computeInitFlowL4(self, imagesL4):
        imagesL3 = self.down4.forward(imagesL4.clone())
        _flowappend = self.up4.forward(self.computeInitFlowL3(imagesL3)) * 2
        
        _img2 = imagesL4[:, 3:6, :, :]
        imagesL4_c = imagesL4.clone()
        
        imagesL4_c[:, 3:6, :, :] = self.image_warp(_img2, _flowappend)
        
        images_in = torch.cat((imagesL4_c, _flowappend), 1)
        flow_est = self.modelL4.forward(images_in)
        return (flow_est + _flowappend)
       
    def computeInitFlowL5(self, imagesL5):
        imagesL4 = self.down5.forward(imagesL5.clone())
        _flowappend = self.up5.forward(self.computeInitFlowL4(imagesL4)) * 2
        
        _img2 = imagesL5[:, 3:6, :, :]
        imagesL5_c = imagesL5.clone()
        
        imagesL5_c[:, 3:6, :, :] = self.image_warp(_img2, _flowappend)
        
        images_in = torch.cat((imagesL5_c, _flowappend), 1)
        flow_est = self.modelL5.forward(images_in)
        return (flow_est + _flowappend)

    
    def makedata(self, img1, img2, flow):
        img1 = Variable(img1).cuda(self.gpu_id)
        img2 = Variable(img2).cuda(self.gpu_id)
        flow = Variable(flow).cuda(self.gpu_id)
        
        height = img1.size(1)
        width = img1.size(2)
        
        imgs = torch.cat((img1, img2), 0).unsqueeze(0)

        if self.level == 1:
            initFlow = Variable(
                torch.zeros(1, 2, height // 16, width // 16).\
                type(torch.FloatTensor)).cuda(self.gpu_id)
            downFlow = self.downs16.forward(flow) / 16.0
            flowDiffOutput = downFlow - initFlow.squeeze(0)
            imgs_d = self.downs16.forward(imgs)

        elif self.level == 2:
            coarseImages = self.downs16.forward(imgs)
            initFlow = self.computeInitFlowL1(coarseImages)
            initFlow = self.ups2.forward(initFlow) * 2.0
            downFlow = self.downs8.forward(flow) / 8.0
            flowDiffOutput = downFlow - initFlow.squeeze(0)
            imgs_d = self.downs8.forward(imgs)

        elif self.level == 3:
            coarseImages = self.downs8.forward(imgs)
            initFlow = self.computeInitFlowL2(coarseImages)
            initFlow = self.ups2.forward(initFlow) * 2.0
            downFlow = self.downs4.forward(flow) / 4.0
            flowDiffOutput = downFlow - initFlow.squeeze(0)
            imgs_d = self.downs4.forward(imgs)

        elif self.level == 4:
            coarseImages = self.downs4.forward(imgs)
            initFlow = self.computeInitFlowL3(coarseImages)
            initFlow = self.ups2.forward(initFlow) * 2.0
            downFlow = self.downs2.forward(flow) / 2.0
            flowDiffOutput = downFlow - initFlow.squeeze(0)
            imgs_d = self.downs2.forward(imgs)

        elif self.level == 5:
            coarseImages = self.downs2.forward(imgs)
            initFlow = self.computeInitFlowL4(coarseImages)
            initFlow = self.ups2.forward(initFlow) * 2.0
            downFlow = flow
            flowDiffOutput = downFlow - initFlow.squeeze(0)
            imgs_d = imgs.clone()

        else:
            pass



        _img2 = imgs_d[:, 3:6, :, :]
        imgs_c = imgs_d.clone()
        
        imgs_c[:, 3:6, :, :] = self.image_warp(_img2, initFlow)
    
        imageFlowInputs = torch.cat((imgs_c, initFlow), 1)

        # the two returned tensors are both size of CHW 3 dimensions
        return imageFlowInputs.squeeze(0).data, flowDiffOutput.data


    def loadTorchModel(self, torch_model_path):
        net_pytorch = Net()
        model_pytorch = net_pytorch.model
        model_torch = load_lua(torch_model_path)

        assert len(model_torch) == len(model_pytorch), 'Error: Torch Module and PyTorch Module have different Lengths!'
        
        for i in range(len(model_torch)):
            if i % 2 == 0:
                model_pytorch[i].weight = torch.nn.Parameter(model_torch.modules[i].weight)
                model_pytorch[i].bias = torch.nn.Parameter(model_torch.modules[i].bias)

        return net_pytorch


    def loadPyTorchModel(self, pytorch_model_path):
        net_pytorch = Net()
        net_pytorch.load_state_dict(torch.load(pytorch_model_path, map_location=lambda storage, loc: storage))
        return net_pytorch