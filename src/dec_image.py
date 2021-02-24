import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderConv(nn.Module):
    def __init__(self, batchnorm_momentum):
        super(DecoderConv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512, momentum=batchnorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=batchnorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=batchnorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=batchnorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0],x.shape[2],1,x.shape[1])
        retval = self.main(x)
        return retval
