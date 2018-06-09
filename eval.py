import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.measure import compare_psnr
import cv2

def show(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
batch_size = 1
learning_rate = 0.0001

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        self.spatempblock = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.BatchNorm3d(64))

        self.layer33 = nn.Sequential(
            nn.Conv3d(64, 256, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU())

        self.layer34 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU())

        self.layer35 = nn.Conv3d(256, 1, kernel_size=(1,3,3), stride=1, padding=(0,1,1))

        self.bn4block = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.BatchNorm3d(64))

        self.conv1x1 = nn.Conv3d(16,1,kernel_size=1,stride=1,padding=0)

    def shuffle(self, fmap):
        #current shape: (N, 16, 3, 128, 128)
        #final shape required: (N, 3, 16, 128, 128)
        N = fmap.shape[0]
        outmap = torch.ones(()).new_empty((N,3,16,128,128))
        outmap[:,0,:,:,:]=fmap[:,:,0,:,:]
        outmap[:,1,:,:,:]=fmap[:,:,1,:,:]
        outmap[:,2,:,:,:]=fmap[:,:,2,:,:]
        #print fmap[:,:,2,:,:]
        return outmap.to(device)

    def anygroup(self, l2):
        #print l2.shape     (N,64,16,128,128)
        outmap = l2.permute(0,2,1,3,4)
        outmap = self.conv1x1(outmap)
        outmap = outmap.permute(0,2,1,3,4)
        #print outmap.shape
        return outmap
        
    def forward(self, x):
        #input dim is (N, C_in, depth, height, width)
        #here we can set it as (1,1,5,128,128)
        l1 = self.layer1(x)
        shuffled = self.shuffle(l1)
        l2 = self.layer2(shuffled)      #size=(N,64,16,128,128)
        l2 = self.anygroup(l2)
        bn4 = self.bn4block(l2)     #size=(N,64,7,128,128)
        out1 = bn4+l2
        bn6 = self.spatempblock(out1)
        out2 = bn6+out1
        bn8 = self.spatempblock(out2)
        out3 = bn8+out2
        bn10 = self.spatempblock(out3)
        out4 = bn10+out3
        bn12 = self.spatempblock(out4)
        out5 = bn12+out4
        bn14 = self.spatempblock(out5)
        out6 = bn14+out5
        bn16 = self.spatempblock(out6)
        out7 = bn16+out6
        bn18 = self.spatempblock(out7)
        out8 = bn18+out7
        bn20 = self.spatempblock(out8)
        out9 = bn20+out8
        bn22 = self.spatempblock(out9)
        out10 = bn22+out9
        bn24 = self.spatempblock(out10)
        out11 = bn24+out10
        bn26 = self.spatempblock(out11)
        out12 = bn26+out11
        bn28 = self.spatempblock(out12)
        out13 = bn28+out12
        bn30 = self.spatempblock(out13)
        out14 = bn30+out13
        bn32 = self.spatempblock(out14)
        out15 = bn32+out14+l2
        #print out15.shape
        l33 = self.layer33(out15)
        l34 = self.layer34(l33)
        l35 = self.layer35(l34)
        return l35

model = ConvNet().to(device)
model.load_state_dict(torch.load("model.ckpt",map_location='cpu'))

X_train=np.load("X_train.npy")
y_train=np.load("y_train.npy")

def compare(val):
    x=X_train[val]; print x.shape
    x=np.array((x[:3,:,:],x[1:4,:,:],x[2:,:,:]))
    x=torch.Tensor(x)
    x=x.unsqueeze(0)
    output=model(x); print "evaluated!"
    img=output.view((128,128))
    img2=((img.data.numpy()+1)/2.0)
    show(np.hstack([(y_train[val]+1)/2.0, (X_train[val,2,:,:]+1)/2.0, img2]))
    imgclear = (y_train[val]+1)/2.0
    imgblur = (X_train[val,2,:,:]+1)/2.0; np.clip(img2, 0, 1, out=img2)
    print "original PSNR: ", compare_psnr(imgblur,imgclear)
    print "output PSNR: ", compare_psnr(img2, imgclear)

for i in range(1140):
    compare(i)
