import torch
import torch.nn as nn
import functools
import torch.utils.data as data
import os
import glob
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from matplotlib import cm
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    return norm_layer


def define_net(input_nc, output_nc, ngf, norm='batch', use_dropout=False, gpu_ids=[], model_name='UNET'):
    """
    Construct a NN model
    Parameters
    ----------
    input_nc : int
        - the number of channels in input images
    output_nc : int
        -- the number of channels in output images
    ngf : int
        -- the number of filters in the last conv layer
    norm : string, optional
        - normalization layer. The default is 'batch'.
    use_dropout : string, optional
       The default is False.
    gpu_ids : TYPE, optional
        DESCRIPTION. The default is [].
    model_name : string, optional
        DESCRIPTION. The default is 'UNET'.

    Returns
    -------
    net : TYPE
        DESCRIPTION.

    """
    net = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

        if model_name == 'UNET':
            net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
        elif model_name == 'RESNET':
            net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
        elif model_name == 'URESNET':
            net = UResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4, gpu_ids=gpu_ids)
        elif model_name == 'URESNET_b':
            net = UResnetGeneratorBoth(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        net.cuda(device=gpu_ids[0])
    net.apply(weights_init)
    return net



class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, my_innermost=False, transposed=True, size=None, add_tanh=True):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if transposed:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
            else:
                upsamp = nn.Upsample(size=size)
                upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
                down = [downconv]
                up = [uprelu, upsamp, upconv]
                if add_tanh:
                    up.append(nn.Tanh())
            model = down + [submodule] + up
        elif innermost:
            if transposed:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
            else:
                upsamp = nn.Upsample(size=size)
                upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upsamp, upconv, upnorm]
            model = down + up
        else:
            mul = 2
            if my_innermost:
                mul = 1
            if transposed:
                upconv = nn.ConvTranspose2d(inner_nc * mul, outer_nc, kernel_size=4, stride=2, padding=1)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
            else:
                upsamp = nn.Upsample(size=size)
                upconv = nn.Conv2d(inner_nc * mul, outer_nc, kernel_size=3, stride=1, padding=1)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upsamp, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
        self.outer_nc = outer_nc
        self.inner_nc = inner_nc

    def forward(self, x):
        if self.outermost:
            out = self.model(x)
        else:
            out = torch.cat([self.model(x), x], 1)
        return out

# Define the corresponding class for your dataset
class my_dataset(data.Dataset):
    def __init__(self, root_path, train_sample_num, val_sample_num, input_file, output1_file, output2_file,  data_mode, train_val_test=0):
        # data_mode: 'vx', 'vy', 'both'
        assert os.path.isdir(root_path), '%s is not a valid directory' % root_path

        # List all JPEG images
        self.den = input_file; #np.load(os.path.join(root_path, 'den.npy'))
 
        self.res = output1_file; #np.load(os.path.join(root_path, 'vx.npy'))
   
        self.res_vy = output2_file; #np.load(os.path.join(root_path, 'vy.npy'))
   
        
        if train_val_test == 0:
            self.den    = self.den[0:train_sample_num]
            self.res    = self.res[0:train_sample_num]
            self.res_vy = self.res_vy[0:train_sample_num]
        elif train_val_test == 1:
            self.den    = self.den[train_sample_num:train_sample_num + val_sample_num]
            self.res    = self.res[train_sample_num:train_sample_num + val_sample_num]
            self.res_vy = self.res_vy[train_sample_num:train_sample_num + val_sample_num]
        elif train_val_test == 2:
            self.den    = self.den[train_sample_num + val_sample_num:]
            self.res    = self.res[train_sample_num + val_sample_num:]
            self.res_vy = self.res_vy[train_sample_num + val_sample_num:]

        self.size = self.den.shape[0]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data_mode = data_mode

        # print('# of samples: {}'.format(self.size))

    def __getitem__(self, index):

        img = self.den[index % self.size]

        img = self.transform(img)                                   # Apply the defined transform

        out = self.res[index % self.size]

        out = self.transform(out)                                   # Apply the defined transform

        if self.data_mode == 'both':
            out_vy = self.res_vy[index % self.size]

            out_vy = self.transform(out_vy)
            out_vy = out_vy.type(torch.FloatTensor)

        img = img.type(torch.FloatTensor)
        out = out.type(torch.FloatTensor)
        if self.data_mode == 'both':
            return img, out, out_vy
        else:
            return img, out

    def __len__(self):
        # Provides the size of the dataset
        return self.size


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []


        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding_mode= "circular", padding=(1,1)
                                 ),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]


        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding_mode= "circular", padding=(1,1)
                                 ),
                       norm_layer(dim),
                       nn.ReLU(True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UResnetGenerator(nn.Module):

#	"""Construct a Unet generator
#        Parameters:
#            input_nc (int)  -- the number of channels in input images
#            output_nc (int) -- the number of channels in output images
#            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
#                                image of size 128x128 will become of size 1x1 # at the bottleneck
#            ngf (int)       -- the number of filters in the last conv layer
#            norm_layer      -- normalization layer
#        We construct the U-Net from the innermost layer to the outermost layer.
#        It is a recursive process.
#        """
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, gpu_ids=[], padding_type='reflect', input_size=128):
        assert(n_blocks >= 0)
        super(UResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        unet_block = nn.Sequential(*model)
        

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8,     unet_block,   my_innermost=True, norm_layer=norm_layer, transposed=False, size=(int(input_size/8 ),int(input_size/8)))
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4,     unet_block,                    norm_layer=norm_layer, transposed=False, size=(int(input_size/4 ),int(input_size/4)))
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,         unet_block,                    norm_layer=norm_layer, transposed=False, size=(int(input_size/2 ),int(input_size/2)))
        unet_block = UnetSkipConnectionBlock(output_nc, ngf,       unet_block, outermost=True,    norm_layer=norm_layer, transposed=False, size=(input_size        ,input_size), add_tanh=False)        # model += [nn.Tanh()]

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UResnetGeneratorBoth(nn.Module):
    #	"""Construct a Unet generator
    #        Parameters:
    #            input_nc (int)  -- the number of channels in input images
    #            output_nc (int) -- the number of channels in output images
    #            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
    #                                image of size 128x128 will become of size 1x1 # at the bottleneck
    #            ngf (int)       -- the number of filters in the last conv layer
    #            norm_layer      -- normalization layer
    #        We construct the U-Net from the innermost layer to the outermost layer.
    #        It is a recursive process.
    #        """


    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', input_size=256):
        assert (n_blocks >= 0)
        super(UResnetGeneratorBoth, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * 32, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        unet_block = nn.Sequential(*model)


        unet_block_vx = UnetSkipConnectionBlock(ngf * 16, ngf * 32, unet_block,  my_innermost=True, norm_layer=norm_layer, transposed=False, size=(int(input_size / 32), int(input_size / 32)))
        unet_block_vx = UnetSkipConnectionBlock(ngf * 8,  ngf * 16, unet_block_vx, norm_layer=norm_layer,                  transposed=False,size=(int(input_size / 16),  int(input_size / 16)))
        unet_block_vx = UnetSkipConnectionBlock(ngf * 4,  ngf * 8 , unet_block_vx, norm_layer=norm_layer,                  transposed=False, size=(int(input_size / 8),  int(input_size / 8)))
        unet_block_vx = UnetSkipConnectionBlock(ngf * 2,  ngf * 4 , unet_block_vx, norm_layer=norm_layer,                  transposed=False, size=(int(input_size / 4),  int(input_size / 4)))
        unet_block_vx = UnetSkipConnectionBlock(ngf,      ngf * 2 , unet_block_vx, norm_layer=norm_layer,                  transposed=False, size=(int(input_size / 2),  int(input_size / 2)))
        unet_block_vx = UnetSkipConnectionBlock(output_nc, ngf, unet_block_vx, outermost=True, norm_layer=norm_layer,      transposed=False, size=(input_size, input_size),    add_tanh=False)

        unet_block_vy = UnetSkipConnectionBlock(ngf * 16, ngf * 32, unet_block, my_innermost=True, norm_layer=norm_layer,  transposed=False, size=(int(input_size / 32), int(input_size / 32)))
        unet_block_vy = UnetSkipConnectionBlock(ngf * 8, ngf * 16,  unet_block_vy, norm_layer=norm_layer, transposed=False,                   size=(int(input_size / 16), int(input_size / 16)))
        unet_block_vy = UnetSkipConnectionBlock(ngf * 4, ngf * 8,   unet_block_vy, norm_layer=norm_layer, transposed=False,                    size=(int(input_size / 8), int(input_size / 8)))
        unet_block_vy = UnetSkipConnectionBlock(ngf * 2, ngf * 4,   unet_block_vy, norm_layer=norm_layer, transposed=False,                    size=(int(input_size / 4), int(input_size / 4)))
        unet_block_vy = UnetSkipConnectionBlock(ngf, ngf * 2,       unet_block_vy, norm_layer=norm_layer, transposed=False,                        size=(int(input_size / 2), int(input_size / 2)))
        unet_block_vy = UnetSkipConnectionBlock(output_nc, ngf,     unet_block_vy, outermost=True, norm_layer=norm_layer,      transposed=False, size=(input_size, input_size),    add_tanh=False)

        self.model_vx = unet_block_vx
        self.model_vy = unet_block_vy

    def forward(self, input):
        return self.model_vx(input), self.model_vy(input)
