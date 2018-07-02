import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from createdata import *

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
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64))

        self.layer33 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.layer34 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.layer35 = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)

        self.bn4block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64))

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
        #print l2.shape
        l2.squeeze_(2)
        bn4 = self.bn4block(l2) 
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
#model.load_state_dict(torch.load("model.ckpt"))

# Loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)

X_train, y_train = createdata()
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

# Train the model
i=0
total_step = 1146
batch_size = 2
for epoch in range(num_epochs):
    i=0
    X_train=np.load("X_train.npy")
    y_train = np.load("y_train.npy") #createdata()
    print "loaded data"
    #np.save("X_train.npy", X_train)
    #np.save("y_train.npy", y_train)
    #exit(0)
    X_train=np.rollaxis(X_train,3,1)        #shape (1146,5,128,128)
    X_train = torch.Tensor(np.array(X_train)/255.0)*2-1
    y_train = torch.Tensor(np.array(y_train)/255.0)*2-1
    for x,y in zip(X_train, y_train):
        i=i+1
        x=np.array((x[:3,:,:].numpy(),x[1:4,:,:].numpy(),x[2:,:,:].numpy()))
        #print x.shape
        x=torch.Tensor(x)
        x=x.permute(1,0,2,3)
        #print x.shape
        x=x.unsqueeze(0)
        x=x.to(device)
        y=y.to(device)
        # Forward pass
        output = model(x)
        output = output.view((3*128*128,1))
        y = y.view((3*128*128,1))
        loss = criterion(output, y)
        print i+1,loss
        #print i+1,loss
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            torch.save(model.state_dict(), 'model.ckpt')
            
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
