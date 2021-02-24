import sys
sys.path.insert(0, '/home/suman/pytorch-caffe')
sys.path.insert(0,'/home/suman/caffe/python/')
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time


def dict_net(protofile, weightfile):
    net = CaffeNet(protofile, width=None, height=None, omit_data_layer=True, phase='TEST')
    net.cuda()
    # print(net)
    net.load_weights(weightfile)
    net.eval()
    return net
    # image = torch.from_numpy(image)
    # if args.cuda:
    #     image = Variable(image.cuda())
    # else:
    #     image = Variable(image)
    # t0 = time.time()
    #blobs = net(image)
    # t1 = time.time()
    #return blobs