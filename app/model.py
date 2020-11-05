import torch as T
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# helper avg_pool function
def pool(out_channels,kernel_size=2, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    pool_layer = nn.AvgPool2d(kernel_size, stride, padding)
    layers.append(pool_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

# residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim,nf1=1,nf2=1):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs

        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3

        self.conv_dim_rslv_layer = conv(in_channels=conv_dim*nf1, out_channels=conv_dim*nf2,
                                kernel_size= 3, stride=1, padding=1)
        self.conv_layer1 = conv(in_channels=conv_dim*nf1, out_channels=conv_dim*nf2,
                                kernel_size= 3, stride=1, padding=1,batch_norm=True)

        self.conv_layer2 = conv(in_channels=conv_dim*nf2, out_channels=conv_dim*nf2,
                               kernel_size= 3, stride=1, padding=1)

    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        y= self.conv_layer2(out_1)
        if x.size()[1]!=y.size()[1]:
           x= self.conv_dim_rslv_layer(x)
        out_2 = x + y
        return out_2

class  GeNet(nn.Module):

    def __init__(self, conv_dim=128, n_res_blocks=4):
        super(GeNet, self).__init__()

        # 1. Define the encoder part of the generator

        # initial convolutional layer given, below
        self.conv1 = conv(1, conv_dim, 4)

        self.avg_pool = pool(2*conv_dim,kernel_size=[2,1],stride=[2,1],padding=0)
        # 2. Define the resnet part of the generator
        # Residual blocks
        res_layers = []

        res_layers.append(pool(conv_dim))
        res_layers.append(ResidualBlock(conv_dim))

        res_layers.append(pool(conv_dim))
        res_layers.append(ResidualBlock(conv_dim))

        res_layers.append(pool(conv_dim))
        res_layers.append(ResidualBlock(conv_dim,nf2=2))

        res_layers.append(pool(2*conv_dim))
        res_layers.append(ResidualBlock(conv_dim,nf1=2,nf2=2))

        # use sequential to create these layers
        self.res_blocks = nn.Sequential(*res_layers)

        fc_layers=[]
        # fully-connected
        fc_layers.append(nn.Linear(768,384))
        fc_layers.append(nn.Dropout2d(p=0.20))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(384,768))
        fc_layers.append(nn.Dropout2d(p=0.20))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(768,10))

        # use sequential to create these layers
        self.fc = nn.Sequential(*fc_layers)

         # drop-out
        self.drop_out= nn.Dropout2d(p=0.20)

    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary

        out = F.relu(self.conv1(x))
        out = F.relu(self.drop_out(self.res_blocks(out)))
        out = self.avg_pool(out)

         # Flatten
        out = out.view(-1, 768)

        # tanh applied to last layer
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


from collections import OrderedDict
class model:
   @staticmethod
   def create(total_classes):
        genet=GeNet()
        layers= OrderedDict()
        mod = list(genet.fc.children())
        mod[-1]=nn.Linear(768,384)
        mod.append(nn.Dropout2d(p=0.40))
        mod.append(nn.ReLU())
        mod.append(nn.Linear(384,192))
        mod.append(nn.Dropout2d(p=0.40))
        mod.append(nn.ReLU())
        mod.append(nn.Linear(192,total_classes))

        layer=['fc','drop_out','relu']
        c=0
        while len(mod)!=1:
          for j,k in zip(layer,mod):
              layers[j+str(c)]=k
          c+=1
          mod=mod[3:]
        else:
          layers['fc'+str(c)]=mod[-1]

        new_fc = nn.Sequential(layers)
        genet.fc=new_fc
        return genet
