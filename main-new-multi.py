

from model1_multi import define_net, my_dataset
import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary


import warnings
warnings.filterwarnings("ignore")

def read_data(var, num_begin, num_end):
    """
    This function reads (loads) the npy file which in this case are vx, vy, and density files.
    
    
    Parameters
    ----------
    var : string
        It represents the identifier in the file name: for example "den" for the density.
    num_begin : int
        Minimum Image ID 
    num_end : int
        Maximum Image ID 
        
        
    Returns
    -------
    Returns a tensor with the size (No. of images, horizontal image size, vertical image size, No. of channels 
                                    for each image i.e. 1 for black and white)
    as well as the minumum and maximum value of the tensor.

    """
    Min=[]
    Max=[]

    den1 = np.load('/home/mo/Documents/LBM/'+var+'.npy')
    Min.append(den1.min())
    Max.append(den1.max())
    den1 = den1[num_begin:num_end]

    Den = den1
    return Den, max(Max), min(Min);
    
#%%
n = 128
den = np.load('/home/mo/Documents/LBM/'+'den'+'.npy')


vx = np.load('/home/mo/Documents/LBM/'+'vx'+'.npy')
vx_min = vx.min()
vx_max = vx.max()

vy = np.load('/home/mo/Documents/LBM/'+'vy'+'.npy')
vy_min = vy.min()
vy_max = vy.max()
den = np.around(den)
vy    = (vy - vy_min)/(vy_max - vy_min)
vx     = (vx - vx_min)/(vx_max - vx_min)

#%%
plt.imshow(den[0,...,0])
plt.imshow(vx[0,...,0])
ratio = 0.00056174e-2
ratio2 = 5e-8
ratioy = 0.00031435e-2
ratio2y = 5e-8


ratio_g = 0

#%%

isTrain = True
model_name = 'URESNET'  # RESNET, UNET, URESNET
input_nc   = 1
output_nc  = 1
gpu_id     = 0
gpu_ids    = [gpu_id]
lr         = 0.0002
tot_var_weight = 0.0
batch_size = 16
Gradientxyx = 0
Gradientxy = 0

ngf        = 8
torch.autograd.set_detect_anomaly(True)
path = '/home/mo/Documents/LBM/weights_new/'
ep_num = 150

train_sample_num = 1200
val_sample_num   = 600

continue_train = False
continue_epoch = 50

netx = define_net(input_nc, output_nc, ngf=ngf, gpu_ids=gpu_ids, model_name=model_name)
nety = define_net(input_nc, output_nc, ngf=ngf, gpu_ids=gpu_ids, model_name=model_name)

Size = 128



if not os.path.exists(path+'weights'):
    os.mkdir(path+'weights')
weights_filename = path+'weights/weights'

if continue_train:
    netx.load_state_dict(torch.load(weights_filename + 'xg_{}.pt'.format(continue_epoch)))
    nety.load_state_dict(torch.load(weights_filename + 'yg_{}.pt'.format(continue_epoch)))

if isTrain:
    old_lr = lr

    # define loss functions

    criterion = torch.nn.L1Loss()
    # initialize optimizers
    optimizery = torch.optim.Adam(list(nety.parameters()) + list(netx.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.00150)

##%% 
dset         = my_dataset(path, train_sample_num, val_sample_num, input_file=den, output1_file=vx, output2_file=vy, data_mode = 'both', train_val_test=0 )


train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)

min_lossx = 1e5
min_lossy = 1e5


train_lossesx = []
train_lossesy = []
val_lossesx = []
val_lossesy = []
tot_varx_2_total = 0
tot_vary_2_total = 0

##%%
for ep in range(ep_num):
    print ('ep #{}'.format(ep))
    train_lossx_total = 0
    train_lossy_total = 0
    netx.train()
    nety.train()
    for img_in, img_out_x, img_out_y in train_loader:
        img_in, img_out_y, img_out_x = Variable(img_in.cuda(gpu_id)),  Variable(img_out_y.cuda(gpu_id)),Variable(img_out_x.cuda(gpu_id))
        
        optimizery.zero_grad()


        recx = netx.forward(img_in)
        recy = nety.forward(img_in)

        recx_grad_x_comp1 = torch.roll(recx, shifts=(0,0,0,-1), dims=(0,0,0,3))
        recx_grad_x_comp2 = torch.roll(recx, shifts=(0,0,0,1), dims=(0,0,0,3))
        recx_grad_x = torch.abs(recx_grad_x_comp1-recx_grad_x_comp2)
        recx_grad_xx = torch.abs(recx_grad_x_comp1-2*recx+recx_grad_x_comp2)
        
        imgoutx_grad_x_comp1 = torch.roll(img_out_x, shifts=(0,0,0,-1), dims=(0,0,0,3))
        imgoutx_grad_x_comp2 = torch.roll(img_out_x, shifts=(0,0,0,1), dims=(0,0,0,3))
        imgoutx_grad_x = torch.abs(imgoutx_grad_x_comp1-imgoutx_grad_x_comp2)
        imgoutx_grad_xx = torch.abs(imgoutx_grad_x_comp1-2*img_out_x+imgoutx_grad_x_comp2)

        
        recx_grad_y_comp1 = torch.roll(recx, shifts=(0,0,-1,0), dims=(0,0,2,0))
        recx_grad_y_comp2 = torch.roll(recx, shifts=(0,0,1,0), dims=(0,0,2,0))
        recx_grad_y = torch.abs(recx_grad_y_comp1-recx_grad_y_comp2)
        recx_grad_yy = torch.abs(recx_grad_y_comp1-2*recx+recx_grad_y_comp2)

        imgoutx_grad_y_comp1 = torch.roll(img_out_x, shifts=(0,0,-1,0), dims=(0,0,2,0))
        imgoutx_grad_y_comp2 = torch.roll(img_out_x, shifts=(0,0,1,0), dims=(0,0,2,0))
        imgoutx_grad_y = torch.abs(imgoutx_grad_y_comp1-imgoutx_grad_y_comp2)
        imgoutx_grad_yy = torch.abs(imgoutx_grad_y_comp1-2*img_out_x+imgoutx_grad_y_comp2)
        
        
        
        tot_varx = (ratio)*(torch.sum(torch.abs(recx_grad_x-imgoutx_grad_x))+torch.sum(torch.abs(recx_grad_y-imgoutx_grad_y)))
        tot_varx_2 = (ratio2)*(torch.sum(torch.abs(recx_grad_xx-imgoutx_grad_xx))+torch.sum(torch.abs(recx_grad_yy-imgoutx_grad_yy)))

        recy_grad_x_comp1 = torch.roll(recy, shifts=(0,0,0,-1), dims=(0,0,0,3))
        recy_grad_x_comp2 = torch.roll(recy, shifts=(0,0,0,1), dims=(0,0,0,3))
        recy_grad_x = torch.abs(recy_grad_x_comp1-recy_grad_x_comp2)
        recy_grad_xx = torch.abs(recy_grad_x_comp1-2*recy+recy_grad_x_comp2)
        
        imgouty_grad_x_comp1 = torch.roll(img_out_y, shifts=(0,0,0,-1), dims=(0,0,0,3))
        imgouty_grad_x_comp2 = torch.roll(img_out_y, shifts=(0,0,0,1), dims=(0,0,0,3))
        imgouty_grad_x = torch.abs(imgouty_grad_x_comp1-imgouty_grad_x_comp2)
        imgouty_grad_xx = torch.abs(imgouty_grad_x_comp1-2*img_out_y+imgouty_grad_x_comp2)

        
        recy_grad_y_comp1 = torch.roll(recy, shifts=(0,0,-1,0), dims=(0,0,2,0))
        recy_grad_y_comp2 = torch.roll(recy, shifts=(0,0,1,0), dims=(0,0,2,0))
        recy_grad_y = torch.abs(recy_grad_y_comp1-recy_grad_y_comp2)
        recy_grad_yy = torch.abs(recy_grad_y_comp1-2*recy+recy_grad_y_comp2)

        imgouty_grad_y_comp1 = torch.roll(img_out_y, shifts=(0,0,-1,0), dims=(0,0,2,0))
        imgouty_grad_y_comp2 = torch.roll(img_out_y, shifts=(0,0,1,0), dims=(0,0,2,0))
        imgouty_grad_y = torch.abs(imgouty_grad_y_comp1-imgoutx_grad_y_comp2)
        imgouty_grad_yy = torch.abs(imgouty_grad_y_comp1-2*img_out_y+imgouty_grad_y_comp2)
        
        
        
        tot_vary = (ratioy)*(torch.sum(torch.abs(recy_grad_x-imgouty_grad_x))+torch.sum(torch.abs(recy_grad_y-imgouty_grad_y)))
        tot_vary_2 = (ratio2y)*(torch.sum(torch.abs(recy_grad_xx-imgouty_grad_xx))+torch.sum(torch.abs(recy_grad_yy-imgouty_grad_yy)))
        

        
        
    

        
        ttvarx = tot_varx.clone()
        ttvarx2 = tot_varx_2.clone()        
        ttvary = tot_vary.clone()
        ttvary2 = tot_vary_2.clone() 
        
        div_loss = ratio2*torch.sum(torch.abs(recx_grad_xx)) + ratio2y*torch.sum(torch.abs(recy_grad_yy))
        gradient_loss = ratio*torch.sum(torch.abs(recx_grad_x)) + ratioy*torch.sum(torch.abs(recy_grad_y))

        recy_loss =  div_loss + gradient_loss + criterion(recy, img_out_y)


        recx_loss =   div_loss + gradient_loss + criterion(recx, img_out_x)

        lossx = recx_loss
        lossx.backward(retain_graph=True)     
        lossy = recy_loss
        lossy.backward()
        optimizery.step()
        
        train_lossx_total += lossx.cpu().data.numpy() * img_in.shape[0]
        train_lossy_total += lossy.cpu().data.numpy() * img_in.shape[0]
        tot_varx_2_total  += tot_varx_2.cpu().data.numpy() * img_in.shape[0]
        tot_vary_2_total  += tot_vary_2.cpu().data.numpy() * img_in.shape[0]
        
    train_lossx_total = train_lossx_total / train_sample_num
    train_lossy_total = train_lossy_total / train_sample_num
    
    tot_varx_2_total = tot_varx_2_total / train_sample_num
    tot_vary_2_total = tot_vary_2_total / train_sample_num

    train_lossesx.append(train_lossx_total)
    train_lossesy.append(train_lossy_total)
    
    
    # Run the model on validation part of the data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    val_dset       = my_dataset(path, train_sample_num, val_sample_num, input_file=den, output1_file=vx, output2_file=vy, data_mode = 'both', train_val_test=1)
    val_loader     = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=True)
    val_lossx_total = 0
    val_lossy_total = 0
    netx.eval()
    nety.eval()
    for img_in, img_out_x, img_out_y in train_loader:
        img_in, img_out_y, img_out_x = Variable(img_in.cuda(gpu_id)),  Variable(img_out_y.cuda(gpu_id)),Variable(img_out_x.cuda(gpu_id))
        recx = netx.forward(img_in)
        recy = nety.forward(img_in)
        
        
 
        recx_grad_x_comp1 = torch.roll(recx, shifts=(0,0,0,-1), dims=(0,0,0,3))
        recx_grad_x_comp2 = torch.roll(recx, shifts=(0,0,0,1), dims=(0,0,0,3))
        recx_grad_x = torch.abs(recx_grad_x_comp1-recx_grad_x_comp2)
        recx_grad_xx = torch.abs(recx_grad_x_comp1-2*recx+recx_grad_x_comp2)
        
        imgoutx_grad_x_comp1 = torch.roll(img_out_x, shifts=(0,0,0,-1), dims=(0,0,0,3))
        imgoutx_grad_x_comp2 = torch.roll(img_out_x, shifts=(0,0,0,1), dims=(0,0,0,3))
        imgoutx_grad_x = torch.abs(imgoutx_grad_x_comp1-imgoutx_grad_x_comp2)
        imgoutx_grad_xx = torch.abs(imgoutx_grad_x_comp1-2*img_out_x+imgoutx_grad_x_comp2)

        
        recx_grad_y_comp1 = torch.roll(recx, shifts=(0,0,-1,0), dims=(0,0,2,0))
        recx_grad_y_comp2 = torch.roll(recx, shifts=(0,0,1,0), dims=(0,0,2,0))
        recx_grad_y = torch.abs(recx_grad_y_comp1-recx_grad_y_comp2)
        recx_grad_yy = torch.abs(recx_grad_y_comp1-2*recx+recx_grad_y_comp2)

        imgoutx_grad_y_comp1 = torch.roll(img_out_x, shifts=(0,0,-1,0), dims=(0,0,2,0))
        imgoutx_grad_y_comp2 = torch.roll(img_out_x, shifts=(0,0,1,0), dims=(0,0,2,0))
        imgoutx_grad_y = torch.abs(imgoutx_grad_y_comp1-imgoutx_grad_y_comp2)
        imgoutx_grad_yy = torch.abs(imgoutx_grad_y_comp1-2*img_out_x+imgoutx_grad_y_comp2)
        
        
        
        tot_varx = (ratio)*(torch.sum(torch.abs(recx_grad_x-imgoutx_grad_x))+torch.sum(torch.abs(recx_grad_y-imgoutx_grad_y)))
        tot_varx_2 = (ratio2)*(torch.sum(torch.abs(recx_grad_xx-imgoutx_grad_xx))+torch.sum(torch.abs(recx_grad_yy-imgoutx_grad_yy)))

        recy_grad_x_comp1 = torch.roll(recy, shifts=(0,0,0,-1), dims=(0,0,0,3))
        recy_grad_x_comp2 = torch.roll(recy, shifts=(0,0,0,1), dims=(0,0,0,3))
        recy_grad_x = torch.abs(recy_grad_x_comp1-recy_grad_x_comp2)
        recy_grad_xx = torch.abs(recy_grad_x_comp1-2*recy+recy_grad_x_comp2)
        
        imgouty_grad_x_comp1 = torch.roll(img_out_y, shifts=(0,0,0,-1), dims=(0,0,0,3))
        imgouty_grad_x_comp2 = torch.roll(img_out_y, shifts=(0,0,0,1), dims=(0,0,0,3))
        imgouty_grad_x = torch.abs(imgouty_grad_x_comp1-imgouty_grad_x_comp2)
        imgouty_grad_xx = torch.abs(imgouty_grad_x_comp1-2*img_out_y+imgouty_grad_x_comp2)

        
        recy_grad_y_comp1 = torch.roll(recy, shifts=(0,0,-1,0), dims=(0,0,2,0))
        recy_grad_y_comp2 = torch.roll(recy, shifts=(0,0,1,0), dims=(0,0,2,0))
        recy_grad_y = torch.abs(recy_grad_y_comp1-recy_grad_y_comp2)
        recy_grad_yy = torch.abs(recy_grad_y_comp1-2*recy+recy_grad_y_comp2)

        imgouty_grad_y_comp1 = torch.roll(img_out_y, shifts=(0,0,-1,0), dims=(0,0,2,0))
        imgouty_grad_y_comp2 = torch.roll(img_out_y, shifts=(0,0,1,0), dims=(0,0,2,0))
        imgouty_grad_y = torch.abs(imgouty_grad_y_comp1-imgoutx_grad_y_comp2)
        imgouty_grad_yy = torch.abs(imgouty_grad_y_comp1-2*img_out_y+imgouty_grad_y_comp2)
        
        
        
        tot_vary = (ratioy)*(torch.sum(torch.abs(recy_grad_x-imgouty_grad_x))+torch.sum(torch.abs(recy_grad_y-imgouty_grad_y)))
        tot_vary_2 = (ratio2y)*(torch.sum(torch.abs(recy_grad_xx-imgouty_grad_xx))+torch.sum(torch.abs(recy_grad_yy-imgouty_grad_yy)))
        
                
        ttvarx = tot_varx.clone()
        ttvarx2 = tot_varx_2.clone()        
        ttvary = tot_vary.clone()
        ttvary2 = tot_vary_2.clone() 
        
        div_loss = ratio2*torch.sum(torch.abs(recx_grad_xx)) + ratio2y*torch.sum(torch.abs(recy_grad_yy))
        gradient_loss = ratio*torch.sum(torch.abs(recx_grad_x)) + ratioy*torch.sum(torch.abs(recy_grad_y))
        
   
        val_lossx =  div_loss + gradient_loss + criterion(recx, img_out_x)

       
        val_lossy =  div_loss + gradient_loss + criterion(recy, img_out_y)

 

        
        
        val_lossx_total += val_lossx.cpu().data.numpy() * img_in.shape[0]
        val_lossy_total += val_lossy.cpu().data.numpy() * img_in.shape[0]
        tot_varx_2_total += tot_varx_2.cpu().data.numpy() * img_in.shape[0]
        tot_vary_2_total += tot_vary_2.cpu().data.numpy() * img_in.shape[0]


    # calculate average validation loss
    val_lossx_total = val_lossx_total / val_sample_num
    val_lossy_total = val_lossy_total / val_sample_num
    
    tot_varx_2_total = tot_varx_2_total / val_sample_num
    tot_vary_2_total = tot_vary_2_total / val_sample_num

    val_lossesx.append(val_lossx_total)
    val_lossesy.append(val_lossy_total)
    # save the model if the new validation loss is less than the previous best
    if val_lossx_total < min_lossx:
        min_lossx = val_lossx_total
        torch.save(netx.state_dict(), weights_filename + str(ratio_g) + 'xg.pt')
        print (weights_filename + '_{}_more.pt is saved for vx!'.format('best'))
    
    if val_lossy_total < min_lossy:
        min_lossy = val_lossy_total
        torch.save(nety.state_dict(), weights_filename + str(ratio_g) + 'yg.pt')
        print (weights_filename + '_{}_more.pt is saved for vy!'.format('best'))

    print ('ValidationX: train loss: {}, val loss: {}'.format(train_lossx_total, val_lossx_total))
    print ('ValidationY: train loss: {}, val loss: {}'.format(train_lossy_total, val_lossy_total))

    plt.figure(figsize=(15, 5))
    
    img_in = np.rollaxis(img_in.cpu().data.numpy(), 1, 4)
    img    = img_in[0]
    plt.subplot(2, 5, 1)
    im = plt.imshow(img[..., 0], cmap='gray')   #, vmin = img1[..., 0].min(), vmax = img1[..., 0].max()       # The values in the plot are bounded between max and min of the predictions (using vmax and vmin) so that we can see the differences in the results
 
    plt.title('raw')

    recx  = recx.cpu().data.numpy()
    recx  = np.rollaxis(recx, 1, 4)
    img1 = recx[0]
    
    plt.subplot(2,5, 2)
    im = plt.imshow(img1[..., 0], cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title('Predicted ep # {}'.format(ep))

    img_out = np.rollaxis(img_out_x.cpu().data.numpy(), 1, 4)
    img     = img_out[0]
    plt.subplot(2, 5, 3)
    im = plt.imshow(img[..., 0], cmap='jet')   #, vmin = img1[..., 0].min(), vmax = img1[..., 0].max()       # The values in the plot are bounded between max and min of the predictions (using vmax and vmin) so that we can see the differences in the results
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title('Truth')

    Img = (img[..., 0] - img1[..., 0]) / (img[..., 0].max() - img[..., 0].min())                                       # This normalization scheme is used. img1 is predictions, img is the truth, Img is the normalized error
    plt.subplot(2, 5, 4)
    im = plt.imshow( Img , cmap='jet')
    plt.title('Error normalized')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    
    a = Img.flatten()
    plt.subplot(2, 5, 5)
    plt.hist(a, bins = 100, density=1, facecolor='green', alpha=0.95);
    plt.title('Error Distribution')

    plt.tight_layout()

    
    
    
    
    
    recy  = recy.cpu().data.numpy()
    recy  = np.rollaxis(recy, 1, 4)
    img1 = recy[0]
    
    #img_in = np.rollaxis(img_in.cpu().data.numpy(), 1, 4)
    img    = img_in[0]
    plt.subplot(2, 5, 6)
    im = plt.imshow(img[..., 0], cmap='gray')   
    plt.title('raw')
    
    plt.subplot(2,5, 7)
    im = plt.imshow(img1[..., 0], cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title('Predicted ep # {}'.format(ep))

    img_out = np.rollaxis(img_out_y.cpu().data.numpy(), 1, 4)
    img     = img_out[0]
    plt.subplot(2, 5, 8)
    im = plt.imshow(img[..., 0], cmap='jet')   #, vmin = img1[..., 0].min(), vmax = img1[..., 0].max()       # The values in the plot are bounded between max and min of the predictions (using vmax and vmin) so that we can see the differences in the results
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.title('Truth')

    Img = (img[..., 0] - img1[..., 0]) / (img[..., 0].max() - img[..., 0].min())                                       # This normalization scheme is used. img1 is predictions, img is the truth, Img is the normalized error
    plt.subplot(2, 5, 9)
    im = plt.imshow( Img , cmap='jet')
    plt.title('Error normalized')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    
    a = Img.flatten()
    plt.subplot(2, 5, 10)
    plt.hist(a, bins = 100, density=1, facecolor='green', alpha=0.95);
    plt.title('Error Distribution')

    plt.tight_layout()
 
    
    
    
    plt.show()

np.savez('div2valtrain.npz', np.array(val_lossesx), np.array(train_lossesx))


np.savez('divtrainlearning_ratemodel1multi_{}.npz'.format(str(ratio)), np.array(train_lossesx), np.array(train_lossesx))


#%%
