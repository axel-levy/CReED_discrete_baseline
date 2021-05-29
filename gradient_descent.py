import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import skimage.io as io
import dataio
import time

# Uncomment to use shapes_6dof
# ************
# filename = 'shapes_6dof'
# duration_tot = 25.
# first_timestamp = 0.
# frame_GT = 'images/frame_00000040.png'
# time_GT = 1.781811999
# duration_reconstruction = 2.
#************

# Uncomment to use slider_depth
# ************
filename = 'slider_depth'
duration_tot = 3.4002690010000003
first_timestamp = 0.003811
frame_GT = 'images/frame_00000018.png'
time_GT = 0.697777000
duration_reconstruction = 1.
# ************

# Uncomment to use office_zigzag
# ************
# filename = 'office_zigzag'
# duration_tot = 10.895518
# first_timestamp = 0.
# frame_GT = 'images/frame_00000017.png'
# time_GT = 0.762878000
# duration_reconstruction = 1.
# ************

event_dataset = dataio.EventsArray('events/'+filename+'/events.npy')

print("Will perform gradient descent on "+str(filename))

W = 180
H = 240
N = 40 # Number of voxels along temporal axis (framerate)

#Ground truth image
img_GT = io.imread('data/' + filename + '/' + frame_GT).astype(float) / 255.
plt.figure(figsize=(10,7),dpi=100)
plt.title("Ground truth frame", fontsize=14)
plt.imshow(img_GT, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

def create_bin_tensor(W,H,n_bins,t_stop = 10):
    bin_tensor = np.zeros((W,H,n_bins),dtype=int)
    bin_size = min(2,(t_stop + 1.))/n_bins
    image = np.zeros((W,H),dtype=float)
    next_bin = -1. + bin_size
    next_t = event_dataset.event_block[0,3]
    n_events = event_dataset.event_block.shape[0]
    c_events = 0
    c_bin = 0
    remaining_events = True
    with tqdm(total=n_bins) as pbar:
        while next_t < t_stop and remaining_events:
            while next_t > next_bin:
                bin_tensor[:,:,c_bin] = image
                next_bin += bin_size
                c_bin += 1
                pbar.update(1)
                image = np.zeros((W,H),dtype=float)
            x = int(event_dataset.event_block[c_events,1])
            y = int(event_dataset.event_block[c_events,2])
            image[x,y] += event_dataset.event_block[c_events,0]
            c_events += 1
            if c_events < n_events:
                next_t = event_dataset.event_block[c_events,3]
            else:
                remaining_events = False
        bin_tensor[:,:,c_bin] = image
        pbar.update(1)
    return(bin_tensor)

# Differential Operators

def temporal_diff(I):
    return(I[:,:,1:]-I[:,:,:-1])

def x_diff(I):
    return(I[1:,:,:]-I[:-1,:,:])

def y_diff(I):
    return(I[:,1:,:]-I[:,:-1,:])

# Event Loss
# 
# $L_e = \frac{1}{WHN}\sum_{x,y,t} (I(x,y,t+1)-I(x,y,t)-b(x,y,t))^2$

def loss_events(I,B):
    return(np.mean((temporal_diff(I)-B)**2))

# TV Regularization
# 
# - Anisotropic case:
# 
# $L_s = \frac{\lambda_s}{WHN}\sum_{x,y,t} |I(x+1,y,t)-I(x,y,t)| + |I(x,y+1,t)-I(x,y,t)|$
# 
# $L_t = \frac{\lambda_t}{WHN}\sum_{x,y,t} |I(x,y,t+1)-I(x,y,t)|$
# 
# - Isotropic case:
# 
#     - Spatio-temporal separation:
#     
#     $L_s = \frac{\lambda_s}{WHN}\sum_{x,y,t} \sqrt{(I(x+1,y,t)-I(x,y,t))^2 + (I(x,y+1,t)-I(x,y,t))^2}$
#     
#     $L_t = \frac{\lambda_t}{WHN}\sum_{x,y,t} |I(x,y,t+1)-I(x,y,t)|$
#     
#     - Spatio-temporal mixing:
#     
#     $L_s = \frac{1}{WHN}\sum_{x,y,t} \sqrt{\lambda_s^2((I(x+1,y,t)-I(x,y,t))^2 + (I(x,y+1,t)-I(x,y,t))^2) + \lambda_t^2(I(x,y,t+1)-I(x,y,t))^2}$

def loss_x(I):
    return(np.sum(np.abs(x_diff(I)))/(I.shape[0]*I.shape[1]*I.shape[2]))

def loss_y(I):
    return(np.sum(np.abs(y_diff(I)))/(I.shape[0]*I.shape[1]*I.shape[2]))

def loss_t(I):
    return(np.sum(np.abs(temporal_diff(I)))/(I.shape[0]*I.shape[1]*I.shape[2]))

def loss_total(I,B,space_reg,time_reg):
    return(loss_events(I,B)+space_reg*(loss_x(I)+loss_y(I))+time_reg*loss_t(I))

def loss_space_iso_separate(I):
    xd2 = x_diff(I)**2
    yd2 = y_diff(I)**2
    return(np.sum(np.sqrt(xd2[:,:-1,:]+yd2[:-1,:,:]))/(I.shape[0]*I.shape[1]*I.shape[2]))

def loss_total_iso_separate(I,B,space_reg,time_reg):
    return(loss_events(I,B)+space_reg*loss_space_iso_separate(I)+time_reg*loss_t(I))

def loss_space_time_iso(I,lambda_s,lambda_t):
    xd2 = x_diff(I)**2
    yd2 = y_diff(I)**2
    td2 = temporal_diff(I)**2
    return(np.sum(np.sqrt(lambda_s**2*(xd2[:,:-1,:-1]+yd2[:-1,:,:-1])+lambda_t**2*td2[:-1,:-1,:]))/(I.shape[0]*I.shape[1]*I.shape[2]))

def loss_total_iso_mixed(I,B,space_reg,time_reg):
    return(loss_events(I,B)+loss_space_time_iso(I,space_reg,time_reg))


# Loss Gradients

def loss_events_gradient(I,B):
    center_gradient = 2.*(-I[:,:,2:] + 2.*I[:,:,1:-1] - I[:,:,:-2] + B[:,:,1:] - B[:,:,:-1])
    left_gradient = 2*(-I[:,:,1] + I[:,:,0] + B[:,:,0])
    right_gradient = 2*(I[:,:,-1] - I[:,:,-2] - B[:,:,-1])
    return(np.concatenate((np.concatenate((left_gradient[:,:,np.newaxis],center_gradient),axis=2),right_gradient[:,:,np.newaxis]),axis=2))

def loss_x_gradient(I):
    center_gradient = np.sign(I[1:-1,:,:] - I[:-2,:,:]) - np.sign(I[2:,:,:] - I[1:-1,:,:])
    left_gradient = - np.sign(I[1,:,:] - I[0,:,:])
    right_gradient = np.sign(I[-1,:,:] - I[-2,:,:])
    return(np.concatenate((np.concatenate((left_gradient[np.newaxis,:,:],center_gradient),axis=0),right_gradient[np.newaxis,:,:]),axis=0))

def loss_y_gradient(I):
    center_gradient = np.sign(I[:,1:-1,:] - I[:,:-2,:]) - np.sign(I[:,2:,:] - I[:,1:-1,:])
    left_gradient = - np.sign(I[:,1,:] - I[:,0,:])
    right_gradient = np.sign(I[:,-1,:] - I[:,-2,:])
    return(np.concatenate((np.concatenate((left_gradient[:,np.newaxis,:],center_gradient),axis=1),right_gradient[:,np.newaxis,:]),axis=1))

def loss_t_gradient(I):
    center_gradient = np.sign(I[:,:,1:-1] - I[:,:,:-2]) - np.sign(I[:,:,2:] - I[:,:,1:-1])
    left_gradient = - np.sign(I[:,:,1] - I[:,:,0])
    right_gradient = np.sign(I[:,:,-1] - I[:,:,-2])
    return(np.concatenate((np.concatenate((left_gradient[:,:,np.newaxis],center_gradient),axis=2),right_gradient[:,:,np.newaxis]),axis=2))

def loss_space_iso_separate_gradient(I):
    xd = x_diff(I)
    yd = y_diff(I)
    inv_sqrt_loss = 1./np.sqrt((xd**2)[:,:-1,:] + (yd**2)[:-1,:,:])
    
    center_gradient_x = -xd[1:,1:-1,:]*inv_sqrt_loss[1:,1:,:] + xd[:-1,1:-1,:]*inv_sqrt_loss[:-1,1:,:]
    left_gradient_x = -xd[0,1:-1,:]*inv_sqrt_loss[0,1:,:]
    right_gradient_x = xd[-1,1:-1,:]*inv_sqrt_loss[-1,1:,:]
    gradient_x = np.concatenate((np.concatenate((left_gradient_x[np.newaxis,:,:],center_gradient_x),axis=0),right_gradient_x[np.newaxis,:,:]),axis=0)
    
    center_gradient_y = -yd[1:-1,1:,:]*inv_sqrt_loss[1:,1:,:] + yd[1:-1,:-1,:]*inv_sqrt_loss[1:,:-1,:]
    left_gradient_y = -yd[1:-1,0,:]*inv_sqrt_loss[1:,0,:]
    right_gradient_y = yd[1:-1,-1,:]*inv_sqrt_loss[1:,-1,:]
    gradient_y = np.concatenate((np.concatenate((left_gradient_y[:,np.newaxis,:],center_gradient_y),axis=1),right_gradient_y[:,np.newaxis,:]),axis=1)
    
    gradient = np.pad(gradient_x,pad_width=((0,0),(1,1),(0,0))) + np.pad(gradient_y,pad_width=((1,1),(0,0),(0,0)))
    
    gradient[0,0,:] = (-xd[0,0,:] - yd[0,0,:])*inv_sqrt_loss[0,0,:]
    gradient[-1,0,:] = xd[-1,0,:]*inv_sqrt_loss[-1,0,:]
    gradient[0,-1,:] = yd[0,-1,:]*inv_sqrt_loss[0,-1,:]
    
    return(gradient)

def loss_space_iso_mixed_gradient(I,lambda_s,lambda_t):
    xd = x_diff(I)
    yd = y_diff(I)
    td = temporal_diff(I)
    inv_sqrt_loss = 1./np.sqrt(lambda_s**2*((xd**2)[:,:-1,:-1] + (yd**2)[:-1,:,:-1])+lambda_t**2*(td**2)[:-1,:-1,:])
    
    center_gradient_x = -xd[1:,1:-1,1:-1]*inv_sqrt_loss[1:,1:,1:] + xd[:-1,1:-1,1:-1]*inv_sqrt_loss[:-1,1:,1:]
    left_gradient_x = -xd[0,1:-1,1:-1]*inv_sqrt_loss[0,1:,1:]
    right_gradient_x = xd[-1,1:-1,1:-1]*inv_sqrt_loss[-1,1:,1:]
    gradient_x = lambda_s**2*np.concatenate((np.concatenate((left_gradient_x[np.newaxis,:,:],center_gradient_x),axis=0),right_gradient_x[np.newaxis,:,:]),axis=0)
    
    center_gradient_y = -yd[1:-1,1:,1:-1]*inv_sqrt_loss[1:,1:,1:] + yd[1:-1,:-1,1:-1]*inv_sqrt_loss[1:,:-1,1:]
    left_gradient_y = -yd[1:-1,0,1:-1]*inv_sqrt_loss[1:,0,1:]
    right_gradient_y = yd[1:-1,-1,1:-1]*inv_sqrt_loss[1:,-1,1:]
    gradient_y = lambda_s**2*np.concatenate((np.concatenate((left_gradient_y[:,np.newaxis,:],center_gradient_y),axis=1),right_gradient_y[:,np.newaxis,:]),axis=1)
    
    center_gradient_t = -td[1:-1,1:-1,1:]*inv_sqrt_loss[1:,1:,1:] + td[1:-1,1:-1,:-1]*inv_sqrt_loss[1:,1:,:-1]
    left_gradient_t = -td[1:-1,1:-1,0]*inv_sqrt_loss[1:,1:,0]
    right_gradient_t = td[1:-1,1:-1,-1]*inv_sqrt_loss[1:,1:,-1]
    gradient_t = lambda_t**2*np.concatenate((np.concatenate((left_gradient_t[:,:,np.newaxis],center_gradient_t),axis=2),right_gradient_t[:,:,np.newaxis]),axis=2)
    
    gradient = np.pad(gradient_x,pad_width=((0,0),(1,1),(1,1))) + np.pad(gradient_y,pad_width=((1,1),(0,0),(1,1))) + np.pad(gradient_t,pad_width=((1,1),(1,1),(0,0)))
    
    gradient[0,0,0] = (lambda_s**2*(-xd[0,0,0] - yd[0,0,0]) - lambda_t**2*td[0,0,0])*inv_sqrt_loss[0,0,0]
    gradient[-1,0,0] = xd[-1,0,0]*inv_sqrt_loss[-1,0,0]
    gradient[0,-1,0] = yd[0,-1,0]*inv_sqrt_loss[0,-1,0]
    gradient[0,0,-1] = td[0,0,-1]*inv_sqrt_loss[0,0,-1]
    
    return(gradient)

def step(I,B,space_reg,time_reg,lr,W,H,N):
    if space_reg>1e-6 and time_reg>1e-6:
        return(I - (lr/(W*H*N))*(loss_events_gradient(I,B) + space_reg*(loss_x_gradient(I) + loss_y_gradient(I)) + time_reg*loss_t_gradient(I)))
    elif space_reg>1e-6:
        return(I - (lr/(W*H*N))*(loss_events_gradient(I,B) + space_reg*(loss_x_gradient(I) + loss_y_gradient(I))))
    elif time_reg>1e-6:
        return(I - (lr/(W*H*N))*(loss_events_gradient(I,B) + time_reg*loss_t_gradient(I)))
    else:
        return(I - (lr/(W*H*N))*(loss_events_gradient(I,B)))

def step_iso_separate(I,B,space_reg,time_reg,lr,W,H,N):
    return(I - (lr/(W*H*N))*(loss_events_gradient(I,B) + space_reg*loss_space_iso_separate_gradient(I) + time_reg*loss_t_gradient(I)))

def step_iso_mixed(I,B,space_reg,time_reg,lr,W,H,N):
    return(I - (lr/(W*H*N))*(loss_events_gradient(I,B) + loss_space_iso_mixed_gradient(I,space_reg,time_reg)))

def explicit_solver(B,W,H,N):
    I = np.zeros((W,H,N),dtype=int)
    with tqdm(total=W*H*(N-1)) as pbar:
        for x in range(W):
            for y in range(H):
                for k in range(1,N):
                    I[x,y,k] = I[x,y,k-1] + B[x,y,k-1]
                    pbar.update(1)
    return(I)

# Quality Metrics

def robust_normalize(img, perc=1):
    the_min = np.percentile(img, perc)
    the_max = np.percentile(img, 100 - perc)
    img_clipped = np.clip(img, a_min=the_min, a_max=the_max)
    return((img_clipped - the_min)/(the_max - the_min))

def compute_MSE(Intensity, time_nd, img_GT):
    img_GT_nd = robust_normalize(img_GT)
    n_frames = Intensity.shape[2]
    idx = int(time_nd * n_frames)
    img = Intensity[:, :, idx]
    the_min = np.percentile(img, 1)
    the_max = np.percentile(img, 99)
    img = np.clip(img, the_min, the_max)
    img_nd = robust_normalize(img)
    return(np.mean((img_nd - img_GT_nd) ** 2))


# Create bin tensor
ratio = duration_reconstruction / duration_tot
t_stop = 2. * ratio - 1.
print("Creating bin tensor...")
Bins = create_bin_tensor(W, H, N-1, t_stop=t_stop)
print("Done")
# In[139]:

time_GT_nd = (time_GT - first_timestamp) / duration_tot / ratio
idx_GT = int(time_GT_nd)

# DEAD PIXEL
Bins[15, 22, :] = 0

# Gradient Descent - Anisotropic TV

### RUN EXPERIMENT ###
n_iter = 100

quick_compute = False

lr = 1e5
space_reg = 0.02
time_reg = 0.

losses_tot = []
losses_events = []
losses_space = []
losses_time = []
MSEs = []
times = [0]

np.random.seed(1)
Intensity = np.random.rand(W,H,N)

Loss_e = loss_events(Intensity,Bins)
Loss_xy = loss_x(Intensity) + loss_y(Intensity)
Loss_t = loss_t(Intensity)
MSE = compute_MSE(Intensity, time_GT_nd, img_GT)

print("Loss Events: "+str(Loss_e))
print("Loss Space: "+str(Loss_xy))
print("Loss Time: "+str(Loss_t))
losses_events.append(Loss_e)
losses_space.append(Loss_xy)
losses_time.append(Loss_t)
MSEs.append(MSE)

tic = time.time()

with tqdm(total=n_iter) as pbar:
    for i in range(n_iter):
        Intensity = step(Intensity,Bins,space_reg,time_reg,lr,W,H,N)
        if not quick_compute:
            Loss_e = loss_events(Intensity,Bins)
            Loss_xy = loss_x(Intensity) + loss_y(Intensity)
            Loss_t = loss_t(Intensity)
            losses_events.append(Loss_e)
            losses_space.append(Loss_xy)
            losses_time.append(Loss_t)
        MSE = compute_MSE(Intensity, time_GT_nd, img_GT)
        MSEs.append(MSE)
        toc = time.time()
        times.append(toc - tic)
        pbar.update(1)

# Show reconstruction
idx = int(time_GT_nd * N)
plt.figure(figsize=(10,7),dpi=100)
plt.imshow(Intensity[:,:,idx],cmap="gray")
plt.title("Reconstructed frame at ground truth time", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.show()

# Plots
plt.figure(figsize=(10,8))
plt.subplot(221)
plt.plot(np.array(losses_events) + space_reg*np.array(losses_space) + time_reg*np.array(losses_time),'k-',label="Total Loss")
plt.legend(loc="best")
plt.yscale("log")
plt.subplot(222)
plt.plot(losses_events,'r-',label="Loss Events")
plt.legend(loc="best")
plt.yscale("log")
if space_reg>0:
    plt.subplot(223)
    plt.plot(space_reg*np.array(losses_space),'g-',label="Loss Space")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.yscale("log")
else:
    plt.subplot(223)
    plt.plot(np.array(losses_space),'g-',label="Loss Space (wo/ factor)")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
if time_reg>0:
    plt.subplot(224)
    plt.plot(time_reg*np.array(losses_time),'b-',label="Loss Time")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.xlabel("Iterations")
else:
    plt.subplot(224)
    plt.plot(np.array(losses_time),'b-',label="Loss Time (wo/ factor)")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
plt.show()


plt.figure(figsize=(8,6), dpi=100)
plt.title("MSE versus Iterations", fontsize=14)
plt.plot(MSEs, 'm-')
plt.yscale('log')
plt.xlabel("Iterations")
plt.show()

plt.figure(figsize=(8,6),dpi=100)
plt.title("MSE versus time", fontsize=14)
plt.plot(times, MSEs, 'm-')
plt.yscale('log')
plt.xlabel("Time (sec)")
plt.show()