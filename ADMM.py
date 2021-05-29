import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import skimage.io as io
import tensorflow as tf
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

print("Will perform ADMM on "+str(filename))

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
    '''Create a bin tensor from event_dataset'''
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

def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy, idz = np.indices(imshape)

    if position == 'center':
        offx, offy, offz = dshape // 2 + 1
    else:
        offx, offy, offz = (0, 0, 0)
        
    pad_img[idx + offx, idy + offy, idz + offz] = image

    return pad_img

def psf2otf(psf, shape, position='corner'):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position=position)

    # Circularly shift OTF so that the 'center' of the PSF is
    # 0 element of the array
    if position == 'corner':
        for axis, axis_size in enumerate(inshape):
            psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    
    psf_tf = tf.convert_to_tensor(psf,dtype=tf.complex64)
    
    # Compute the OTF
    otf_tf = tf.signal.fft3d(psf_tf)
    
    proto_tensor = tf.make_tensor_proto(otf_tf)
    otf = tf.make_ndarray(proto_tensor)

    return otf

# Create bin tensor
ratio = duration_reconstruction / duration_tot
t_stop = 2. * ratio - 1.
print("Creating bin tensor...")
Bins = create_bin_tensor(W, H, N-1, t_stop=t_stop)
print("Done")

time_GT_nd = (time_GT - first_timestamp) / duration_tot / ratio
idx_GT = int(time_GT_nd)

# DEAD PIXEL
Bins[15, 22, :] = 0

# Circular padding for Neumann boundary condition in time
n_padding_t = 0
Bins = np.pad(Bins,pad_width=((0,0),(0,0),(n_padding_t,n_padding_t+1)))
right_tensor = -Bins[:,:,::-1]
Bins = np.concatenate((Bins,right_tensor),axis=2)
Nt = Bins.shape[2]
Bins = -Bins

# Gradient kernels and Fourier transforms
dx = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,-1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]]])
dy = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,-1,0],[0,1,0]],[[0,0,0],[0,0,0],[0,0,0]]])
dt = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,-1,1],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])

print("Computing for x...")
shape = (W,H,Nt)
dxFT = psf2otf(dx, shape)
dxFT_tf = tf.convert_to_tensor(dxFT,dtype=tf.complex64)
dxTFT_tf = tf.math.conj(dxFT_tf)
print("Computing for y...")
dyFT = psf2otf(dy, shape)
dyFT_tf = tf.convert_to_tensor(dyFT,dtype=tf.complex64)
dyTFT_tf = tf.math.conj(dyFT_tf)
print("Computing for t...")
shape_t = (W,H,Nt)
dtFT = psf2otf(dt, shape_t)
dtFT_tf = tf.convert_to_tensor(dtFT,dtype=tf.complex64)
dtTFT_tf = tf.math.conj(dtFT_tf)
print("Done")

def spatial_gradient(I_tf):
    F_I_tf = tf.signal.fft3d(I_tf)
    Dx_I_tf = tf.expand_dims(tf.math.real(tf.signal.ifft3d(tf.math.multiply(dxFT_tf,F_I_tf))),axis=3)
    Dy_I_tf = tf.expand_dims(tf.math.real(tf.signal.ifft3d(tf.math.multiply(dyFT_tf,F_I_tf))),axis=3)
    return(tf.cast(tf.concat([Dx_I_tf,Dy_I_tf],axis=3),dtype=tf.complex64))

def update_x_TV(v, numerator_x_1_tf, denominator_x_tf, rho):
    numerator_x_2_tf = rho*(tf.math.multiply(dxTFT_tf,tf.signal.fft3d(v[:,:,:,0])) + tf.math.multiply(dyTFT_tf,tf.signal.fft3d(v[:,:,:,1])))
    new_I_tf = tf.math.real(tf.signal.ifft3d(tf.math.divide((numerator_x_1_tf + numerator_x_2_tf),denominator_x_tf)))
    return(tf.cast(new_I_tf,dtype=tf.complex64))

def update_z_TV(v,kappa):
    v_float = tf.cast(v,dtype=tf.float64)
    new_z_tf = tf.math.maximum(v_float-kappa,0.) - tf.math.maximum(-v_float-kappa,0.)
    return(tf.cast(new_z_tf,dtype=tf.complex64))

def get_residual(I_tf, B, Dx_tf, lam):
    F_I_tf = tf.signal.fft3d(I_tf)
    Dt_I_tf = tf.math.real(tf.signal.ifft3d(tf.math.multiply(dxFT_tf,F_I_tf)))
    proto_tensor = tf.make_tensor_proto(Dt_I_tf)
    Dt_I = tf.make_ndarray(proto_tensor).astype(np.float32)
    proto_tensor = tf.make_tensor_proto(Dx_tf)
    Dx = tf.make_ndarray(proto_tensor).astype(np.float32)
    return(0.5 * np.sum((Dt_I - B) ** 2) + lam * np.sum(np.abs(Dx)))

def show_frame(Intensity_tf, time_nd):
    proto_tensor = tf.make_tensor_proto(Intensity_tf)
    Intensity = tf.make_ndarray(proto_tensor).astype(np.float32)
    n_frames = Intensity.shape[2] // 2
    idx = int(time_nd * n_frames)
    plt.figure(figsize=(10,7),dpi=100)
    plt.imshow(Intensity[:,:,idx],cmap='gray')
    plt.title("Reconstructed frame at ground truth time", fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def show_frame_idx(Intensity, idx):
    plt.figure(figsize=(10, 7), dpi=100)
    plt.imshow(Intensity[:,:,idx],cmap='gray')
    plt.show()

def robust_normalize(img, perc=1):
    the_min = np.percentile(img, perc)
    the_max = np.percentile(img, 100 - perc)
    img_clipped = np.clip(img, a_min=the_min, a_max=the_max)
    return((img_clipped - the_min)/(the_max - the_min))
    
def compute_MSE(Intensity_tf, time_nd, img_GT):
    img_GT_nd = robust_normalize(img_GT)
    proto_tensor = tf.make_tensor_proto(Intensity_tf)
    Intensity = tf.make_ndarray(proto_tensor).astype(np.float32)
    n_frames = Intensity.shape[2] // 2
    idx = int(time_nd * n_frames)
    img = Intensity[:, :, idx]
    the_min = np.percentile(img, 1)
    the_max = np.percentile(img, 99)
    img = np.clip(img, the_min, the_max)
    img_nd = robust_normalize(img)
    return(np.mean((img_nd - img_GT_nd) ** 2))


def ADMM_TV_aniso(I0,B,rho,lam,maxiter, quick_compute=False):
    I = np.copy(I0)
    I_tf = tf.convert_to_tensor(I,dtype=tf.complex64)
    z_tf = spatial_gradient(I_tf)
    u = np.zeros((*I.shape,2),dtype=np.complex64)
    u_tf = tf.convert_to_tensor(u,dtype=tf.complex64)
    B_tf = tf.convert_to_tensor(B,dtype=tf.complex64)
    residuals = []
    MSEs = []
    times = [0]
    
    Dx_tf = spatial_gradient(I_tf)
    new_r = get_residual(I_tf, B, Dx_tf, lam)
    print("Residual = "+str(new_r))
    residuals.append(new_r)
    
    MSE = compute_MSE(I_tf, time_GT_nd, img_GT)
    print('MSE: '+str(MSE))
    MSEs.append(MSE)
    
    print("Pre-computing denominator...")
    denominator_x_tf = tf.math.multiply(dtTFT_tf,dtFT_tf) + rho*(tf.math.multiply(dxTFT_tf,dxFT_tf) + tf.math.multiply(dyTFT_tf,dyFT_tf))
    print("Pre-computing numerator...")
    numerator_x_1_tf = tf.math.multiply(dtTFT_tf,tf.signal.fft3d(B_tf))
    
    tic = time.time()
    
    for i in range(maxiter):
        print("Iteration "+str(i+1)+"/"+str(maxiter))
        I_tf = update_x_TV(z_tf - u_tf, numerator_x_1_tf, denominator_x_tf, rho)
        Dx_tf = spatial_gradient(I_tf)
        z_tf = update_z_TV(Dx_tf + u_tf,lam/rho)
        u_tf = u_tf + Dx_tf - z_tf
        
        if not quick_compute:
            new_r = get_residual(I_tf, B, Dx_tf, lam)
            print("Loss: "+str(new_r))
            residuals.append(new_r)
            
            # show_frame(I_tf, time_GT_nd)
        
        MSE = compute_MSE(I_tf, time_GT_nd, img_GT)
        print('MSE: '+str(MSE))
        MSEs.append(MSE)
        
        toc = time.time()
        
        times.append(toc - tic)

    return(I_tf, residuals, MSEs, times)


### RUN EXPERIMENT ###
maxiter = 10
rho = 1
lam = 0.01
I0 = np.random.rand(W, H, Nt)
quick_compute = False

Intensity_tf, residuals, MSEs, times = ADMM_TV_aniso(I0, Bins, rho, lam, maxiter, quick_compute=quick_compute)

# Show reconstruction
show_frame(Intensity_tf, time_GT_nd)

# Plots
plt.figure(figsize=(8,6), dpi=100)
plt.title("Loss versus iteration", fontsize=14)
plt.plot(residuals, 'r-')
plt.yscale('log')
plt.xlabel("Iterations")
plt.show()

plt.figure(figsize=(8,6), dpi=100)
plt.title("Loss versus time", fontsize=14)
plt.plot(times, residuals, 'r-')
plt.yscale('log')
plt.xlabel("Time (sec)")
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