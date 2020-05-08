# full assembly of the sub-parts to form the complete net

from .unet_parts import *
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        self.up1  = up(1024, 256)
        self.up2  = up(512,  128)
        self.up3  = up(256,  64)
        self.up4  = up(128,  64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        #######################################################################
        # finish your code here
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out
        
        #######################################################################
        
        # After you finish your code above remember to delete the following pass
        # pass

class UNet_shallow(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_shallow, self).__init__()
        self.n_classes = n_classes
        #######################################################################
        # finish your code here 
        # initial the component here
        self.n_classes = n_classes
        
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 256)
        self.up1  = up(384,  128)
        self.up2  = up(192,  64)
        self.outc = outconv(64, n_classes)
        #######################################################################
        


    def forward(self, x):
        #######################################################################
        # finish your code here
        x1 = self.inc(x) # 64*128*128
        x2 = self.down1(x1) # 128*64*64
        x3 = self.down2(x2) # 256*32*32
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        out = self.outc(x)
        return out
        
        #######################################################################
        # After you finish your code above remember to delete the following pass
        # pass


class UNet_add(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_add, self).__init__()
        self.n_classes = n_classes
        
        self.inc   = inconv(n_channels, 64)
        self.down1 = down(64,  128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        
        #####################################################################
        # finish your code here
        self.up1  = up_add(512, 256)
        self.up2  = up_add(256, 128)
        self.up3  = up_add(128, 64)
        self.up4  = up_add(64,  64)
        self.outc = outconv(64, n_classes)
        #####################################################################


    def forward(self, x):
        #######################################################################
        # finish your code here
        x1 = self.inc(x) # 64*128*128
        x2 = self.down1(x1) # 128*64*64
        x3 = self.down2(x2) # 256*32*32
        x4 = self.down3(x3) # 512*16*16
        x5 = self.down4(x4) # 512*8*8
        x = self.up1(x5, x4) # 256*16*16
        x = self.up2(x, x3) # 128*32*32
        x = self.up3(x, x2) # 64*64*64
        x = self.up4(x, x1) # 64*128*128
        out = self.outc(x)
        return out
        
        #######################################################################
        
        # After you finish your code above remember to delete the following pass
        # pass