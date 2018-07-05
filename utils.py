import os
import glob
import struct
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

to_tensor = transforms.ToTensor()


def loadFlow(filename, height, width):
    f = open(filename, 'rb')

    tag, w, h = struct.unpack('fii', f.read(4 + 4 + 4))
    assert tag == 202021.25, 'Error: unable to read .flo file perhaps bigendian error'

    flow = torch.from_numpy(((np.fromfile(f, dtype=np.float32)).\
                                reshape(height, width, 2)).transpose(2, 0, 1))

    f.close()
    return flow


def loadImage(filename):
    img = to_tensor(Image.open(filename))
    return img