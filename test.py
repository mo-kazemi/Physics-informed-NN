import os
os.chdir('/home/mo/Downloads/palabos/ML/practice')

from model1 import define_net, my_dataset
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
#from   torchsummary import summary
import seaborn as sns
import matplotlib.mlab as mlab
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

#%%
den = np.load('/home/mo/Downloads/palabos/ML/practice/'+'den'+'.npy')
den_min = den.min()
den_max = den.max()

vx = np.load('/home/mo/Downloads/palabos/ML/practice/'+'vx'+'.npy')
vx_min = vx.min()
vx_max = vx.max()

den = np.around(den)
vx     = (vx - vx_min)/(vx_max - vx_min)
#%%
model_name = 'URESNET'  # RESNET, UNET
input_nc  = 1
output_nc = 1
gpu_id    = 0
gpu_ids   = [gpu_id]
ngf       = 8
batch_size = 16
# path = '/media/hadi/HHD 6TB/Datasets/Oil/test-input'
path = '/home/mo/Downloads/palabos/ML/practice/'

train_sample_num = 50
val_sample_num   = 0 

net = define_net(input_nc, output_nc, ngf=ngf, gpu_ids=gpu_ids, model_name=model_name)

#summary(net, (1, 256, 256))

weights_filename = path+'weights/weights'
#print(next(net.parameters()).device)

ratio = 5e-6
ratio2 = 1e-5

#%%
#net.load_state_dict(torch.load(weights_filename + str(ratio) + '.pt', map_location='cuda:0'))
net.load_state_dict(torch.load('/home/mo/Downloads/palabos/ML/practice/weights/weights0.0001.pt', map_location='cuda:0'))


dset         = my_dataset(path, train_sample_num, val_sample_num, input_file=den, output1_file=vx, train_val_test=2)           #(3550 - 3000 - 540)
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
#%%
net.eval()
i = 0
trim = 0
trim2 = 128
fig, ax = plt.subplots(10, 5, figsize=(15,30), squeeze=True, gridspec_kw = dict(bottom=0))
for img_in, img_out in train_loader:
    img_in, img_out = Variable(img_in.cuda(gpu_id)), Variable(img_out.cuda(gpu_id))
    print(img_in.shape, img_out.shape)
    rec             = net.forward(img_in).cpu().data.numpy()
    # rec = img_in.cpu().data.numpy()
    rec = np.rollaxis(rec, 1, 4) 

    img_in          = np.rollaxis(img_in.cpu().data.numpy(), 1, 4)
    img_out         = np.rollaxis(img_out.cpu().data.numpy(), 1, 4)
    
    for test_num in range(16):
        j=0

        Img_in = img_in[test_num]
        im = ax[i,j].imshow(Img_in[trim:trim2,trim:trim2, 0], cmap='gray')
        ax[i,j].set_title('Input')
        cbar1 = fig.colorbar(im, ax=ax[i,j], fraction=0.046, pad=0.04)
        
        img1 = rec[test_num]
        im = ax[i,j+1].imshow(img1[trim:trim2,trim:trim2, 0], cmap='jet')
        ax[i,j+1].set_title('Predicted')
        cbar2 = fig.colorbar(im, ax=ax[i,j+1], fraction=0.046, pad=0.04)
        ax[i,j+1].axis('off')

        img = img_out[test_num]
        im = ax[i,j+2].imshow(img[trim:trim2,trim:trim2, 0], cmap='jet')#, vmin = img1[2:-2,2:-2, 0].min(), vmax = img1[2:-2,2:-2, 0].max())
        #im.axes.get_yaxis().set_visible(False)
        cbar1 = fig.colorbar(im, ax=ax[i,j+2], fraction=0.046, pad=0.04)
        ax[i,j+2].set_title('LBS results')
        ax[i,j+2].axis('off')
        
        Img =(img[trim:trim2,trim:trim2, 0] - img1[trim:trim2,trim:trim2, 0]) / (img[trim:trim2,trim:trim2, 0].max() - img[trim:trim2,trim:trim2, 0].min())     # This normalization scheme is used. img1 is predictions, img is the truth, Img is the normalized error
        im = ax[i,j+3].imshow( Img , cmap='jet')
        ax[i,j+3].set_title('Error ')
        cbar1 = fig.colorbar(im, ax=ax[i,j+3], fraction=0.046, pad=0.04)
        ax[i,j+3].axis('off')
        
        # a = Img.flatten()
        # ax[i,j+4].hist(a, bins = 100, density=1, facecolor='green', alpha=0.95);
        ax[i,j+4].plot(img1[trim:trim2,64, 0])
        ax[i,j+4].plot(img[trim:trim2,64, 0])

        ax[i,j+4].set_title('Error Distribution')
        #ax[i,j+4].set_xlim(-.2,.2)
               
        fig.tight_layout()
        print('i: {}, Truth: {}, Predicted: {}'.format(i, np.mean(img[..., 0]), np.mean(img1[..., 0])))
        i=i+1
        

        
    #plt.savefig('error.png', bbox_inches='tight')
#%%        Histograms
# i = 0
# fig, ax = plt.subplots(4, 2, figsize=(15,9), squeeze=True, gridspec_kw = dict(bottom=0))
# for img_in, img_out in train_loader:
#     img_in, img_out = Variable(img_in.cuda(gpu_id)), Variable(img_out.cuda(gpu_id))

#     rec1             = net.forward(img_in).cpu().data.numpy()
#     # rec = img_in.cpu().data.numpy()
#     rec = np.rollaxis(rec1, 1, 4) 
    
#     img_in          = np.rollaxis(img_in.cpu().data.numpy(), 1, 4)
#     img_out         = np.rollaxis(img_out.cpu().data.numpy(), 1, 4)
    

#     for test_num in range(4):
#         j=0
        
#         Img_out = img_out[test_num]
#         sns.distplot(Img_out[..., 0].flatten(), kde=True, color="b", ax=ax[i,j])
#         img1 = rec[test_num]
#         sns.distplot(img1[..., 0].flatten(), kde=True, color="r", ax=ax[i,j+1])
#         ax[i,j].set_title('Input')
        
        
#        img1 = rec[test_num]
#        im = ax[i,j+1].imshow(img1[..., 0], cmap='jet')
#        ax[i,j+1].set_title('Predicted')
#        
#
#
#        
#        img = img_out[test_num]
#        im = ax[i,j+2].imshow(img[..., 0], cmap='jet', vmin = img1[..., 0].min(), vmax = img1[..., 0].max())
#        #im.axes.get_yaxis().set_visible(False)
#        ax[i,j+2].set_title('LBS results')
#        
#        
#
#        Img =(img[..., 0] - img1[..., 0]) / (img[..., 0].max() - img[..., 0].min())     # This normalization scheme is used. img1 is predictions, img is the truth, Img is the normalized error
#        im = ax[i,j+3].imshow( Img , cmap='jet')
#        ax[i,j+3].set_title('Error ')
#        
#        
#        a = Img.flatten()
#        ax[i,j+4].hist(a, bins = 100, density=1, facecolor='green', alpha=0.95);
#        ax[i,j+4].set_title('Error Distribution')
#        ax[i,j+4].set_xlim(-.2,.2)
        

        # fig.tight_layout()
        # i=i+1 
    
    
#%%

img_in_temp_x = img_in[:, :, :, :-1]*img_in[:, :, :, 1:]
img_in_temp_y = img_in[:, :, :-1, :]*img_in[:, :, 1:, :]


# tot_var = (ratio)* ((torch.sum(torch.abs(rec[:, :, :, :-1] - rec[:, :, :, 1:])*img_in_temp_x)) +
                # (torch.sum(torch.abs(rec[:, :, :-1, :] - rec[:, :, 1:, :])*img_in_temp_y)))
tot_var = (ratio)*(                
                        (np.sum((rec[:, :, :, :-1]     - rec[:, :, :, 1:]    )*img_in_temp_x-
                                    (img_out[:, :, :, :-1] - img_out[:, :, :, 1:])*img_in_temp_x)) +
                        (np.sum((rec[:, :, :-1, :]     - rec[:, :, 1:, :]    )*img_in_temp_y-
                                    (img_out[:, :, :-1, :] - img_out[:, :, 1:, :])*img_in_temp_y))
                        )    
    
    
    
#%%
######################################Perm measurer
#del(D, C)
#%%
test_num = 12

pred = []
lbs = []    
for img_in, img_out in train_loader:
    img_in, img_out = Variable(img_in.cuda(gpu_id)), Variable(img_out.cuda(gpu_id))

    rec = net.forward(img_in).cpu().data.numpy()
    rec = np.rollaxis(rec, 1, 4)
    
    img = np.rollaxis(img_out.cpu().data.numpy(), 1, 4)
    
    for Test_num in range(batch_size):        
        img1 = rec[Test_num]        
        Img  = img[Test_num]
        
        pred.append(img1.mean())
        lbs.append( Img.mean())
        print('Truth: {}, Predicted: {}'.format(np.mean(Img), np.mean(img1)))
    C = pd.DataFrame( pred, columns=['Predicted'])
    D = pd.DataFrame( lbs, columns=['LBS Results'])
        
#    
    D = pd.concat([D,C], axis=1)

#%%
sns.jointplot(x=D['Predicted'], y=D['LBS Results'])
plt.savefig('cross-plot.png',bbox_inches='tight')
#%%   plot slice
np.savez('comparison.npz', rec[-1], img_out[-1])
