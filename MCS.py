# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 12:01:27 2020

Script for MCS deconvolution of quasar data.

@author: Giovanni D'Addario & Diganta Bandopdhyay
"""

import numpy as np
import scipy.fftpack as fp
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from astropy.io import fits
from skimage import restoration


'''Set up figure style'''

fig_params = {
   'figure.autolayout': True,
   'font.size': 17.0,
   'savefig.format': 'pdf'
   }
mpl.rcParams.update(fig_params)


'''Import data, crop to relevant part and normalise'''

name = 'stacked1.fits' # get data from current directory
with fits.open(name) as Fits:
    fitsdata = Fits[0].data
    
fitsdata = fitsdata[56:75,11:30] # crop data to 19 x 19 frame
fitsdata = fitsdata/np.max(fitsdata) # normalise


'''Useful functions'''

def deconvolve(x_t,z_t,ifftshift = False):
    '''
    Assumes a convolution x(t) = (y*z)(t): works out y(k)=x(k)/z(k) and 
    recasts it to y(t).
    
    Parameters:
    x(t) : array 
    z(t) : array 
    
    Returns:
    y(t) : array
    '''
    epsilon = 10**(-2)
    x_k = fp.fft2(x_t)
    if ifftshift == True:
        z_k = fp.fft2(fp.ifftshift(z_t))
    else:
        z_k = fp.fft2(z_t)
    freq_kernel = 1/(epsilon+z_k)
    F = x_k*freq_kernel
    Z = fp.ifft2(F).real
    return(Z)


def reconvolve(x_t,z_t,ifftshift = False):
    '''
    Assumes a convolution y(t) = (x*z)(t): works out y(k)=x(k)z(k) and 
    recasts it to y(t).
        
    Parameters:
    x(t) : array 
    z(t) : array 
    
    Returns:
    y(t) : array
    '''
    x_k = fp.fft2(x_t)
    if ifftshift == True:
        z_k = fp.fft2(fp.ifftshift(z_t))
    else:
        z_k = fp.fft2(z_t)
    freq_kernel = z_k
    F = x_k*freq_kernel
    Z = fp.ifft2(F).real
    return(Z)    

    
def circle_PSF(r, shape):
    '''
    Defines a frequency mask in Fourier space.
    
    Parameters:
    r : radius of mask
    shape : list with shape [x, y]
    
    Returns:
    result : mask
    '''
    result = np.zeros(shape)
    for i in range(int(-shape[0]/2),int(shape[0]/2)):
        for j in range(int(-shape[1]/2),int(shape[1]/2)):
            if (((i-shape[0]/2)**2 + (j-shape[1]/2)**2 )<= r**2) or (
                ((i+shape[0]/2)**2 + (j+shape[1]/2)**2 )<= r**2) or (
                ((i-shape[0]/2)**2 + (j+shape[1]/2)**2 )<= r**2) or (
                ((i+shape[0]/2)**2 + (j-shape[1]/2)**2 )<= r**2):
                result[i,j]=1
    return(result)


def extract_background_fourier_median(image, r):
    '''
    Extracts background using Fourier space thresholding
    
    Parameters:
    image : data
    r : radius of frequency mask
    
    '''
    image2 = fp.fft2(image)
    image3 = image2*circle_PSF(r, [19,19])
    return(fp.ifft2(image3).real)


def shift_1D(PSF, offset):
    '''
    Shifts point spread functions by a given offset.
    
    Parameters:
    PSF :  1d array of length N, where N is number of pixels
    offset : tuple (x, y) of offsets , where x and y are 1d arrays of length H
    
    Returns:
    shifted_PSFs : 2d array, each row is a version of the original PSF shifted 
        by one of the offsets
    '''
    N = len(PSF)
    n = int(np.sqrt(N))
    H = len(offset[0])
    shifted_PSFs = np.zeros((H, N)) # each row is a shifted psf for a given
        # offset: create array to store all psfs
    offset_num = np.arange(H)
    pixels = np.arange(N)
    x_pix = pixels % (n)
    y_pix = pixels // (n)
    for num in offset_num:
        for pix in pixels:
            if (x_pix[pix] + offset[0][num] >= 0 ) and (
                y_pix[pix] + offset[1][num] >=0) and (
                x_pix[pix] + offset[0][num] <= n-1) and (
                y_pix[pix] + offset[1][num] <= n-1):
                x_pix_new = (x_pix[pix] + offset[0][num]) % (n)
                y_pix_new = (y_pix[pix] + offset[1][num]) % (n)
                shifted_PSFs[num, x_pix_new + y_pix_new*n] = PSF[pix]
            else:
                pass
    return shifted_PSFs

'''Set up point spread functions and initial guesses/params'''

# fwhm and std dev
FWHM = 2   
std_dev = FWHM/(2.35)  # FWHM = 2.35 std_dev; minimum limit is 2 pix per FWHM

#PSFs: naming conventions match report
T = np.outer(signal.gaussian(19, 2),signal.gaussian(19, 2))
# narrower one
R = (np.outer(signal.gaussian(19, std_dev), signal.gaussian(19, std_dev)))
S = deconvolve(T, R,ifftshift=True)

# modify shape of PSFs to 1d array
R = R.reshape(361)
S = S.reshape(361)

# get all possible offsets for a 19x19 pixel frame
offsetx = [] # x direction
offsety = [] # y direction
for i in np.arange(-9,10):
    for j in np.arange(-9,10):
        offsetx.append(i)
        offsety.append(j)           
offsets = (np.array(offsetx),np.array(offsety))

# get shifted version of R and S for all the offsets above
r = shift_1D(R, offsets) 
s = shift_1D(S, offsets)

# deconvolved data
F = deconvolve(fitsdata  -  np.median(fitsdata,axis=0), S.reshape(19, 19), 
               ifftshift=True)
f = np.reshape(F, (361))

# other variables; naming conventions match report
c = np.array([298, 332, 308, 64]) # positions (1,2,3,4) see lab book 1-12-19
N = len(S) # number of pixels
M = 4 # number of point sources in frame
I = np.identity(N) # N x N identity matrix
global lm
lm = 0.0001 # Lagrange multiplier lambda


'''Define phi and its derivatives in matrix form'''

def phi(h, s, lm, tau, r, I):
    '''
    Defines function which needs to be minimised with respect to the noise
    distribution in the frame. All naming conventions match the report.
    
    Parameters:
    h : noise distribution
    s : shifted versions of S
    lm : Lagrangian multiplier
    tau : float
    r : shifted versions of r
    I : identity matrix
    
    Returns:
    phi : float
    '''
    phi = np.sum((1/N)*((s@(h + tau)-f).T) @ (s@(h + tau)-f) + lm*(
            (h - r @ h).T)@(h - r @ h))
    return phi


def dphi_dh(h, s, lm, tau, r, I):
    '''
    Defines first derivative of phi with resepct to vector h. All naming
    conventions match the report.
    
    Parameters:
    h : noise distribution
    s : shifted versions of S
    lm : Lagrangian multiplier
    tau : float
    r : shifted versions of r
    I : identity matrix
    
    Returns:
    first_deriv : 1d array
    '''
    first_deriv =  (2/N) *(s @ (s @ (h - tau) - f)) + 2*lm*(np.linalg.matrix_power((I-r), 2) @ h)
    return first_deriv


def d2phi_dh2(s, lm, r, I):
    '''
    Defines second derivative of phi with resepct to vector h. All naming
    conventions match the report.
    
    Parameters:
    h : noise distribution
    s : shifted versions of S
    lm : Lagrangian multiplier
    tau : float
    r : shifted versions of r
    I : identity matrix
    
    Returns:
    second_deriv : 2d array
    '''
    second_deriv = (2/N) *(np.linalg.matrix_power(s, 2)) + 2*lm*np.linalg.matrix_power((I-r), 2)
    return second_deriv


'''Improve estimate of h iteratively'''

def new_h(h, s, lm, tau, r, I):
    '''
    Obtains an improved estimate of the error distribution vector h according
    to the formula in the report.
    
    Parameters:
    h : noise distribution
    s : shifted versions of S
    lm : Lagrangian multiplier
    tau : float
    r : shifted versions of r
    I : identity matrix
    
    Returns:
    new_h : 1d array
    '''
    new_h = h - d2phi_dh2(s, lm, r, I) @ dphi_dh(h, s, lm, tau, r, I)
    return new_h


def iterate_new_h(h, s, lm, tau, r, I, turns):
    '''
    Execute several iterations of procedure for improving error estimate.
    
    Parameters:
    h : noise distribution
    s : shifted versions of S
    lm : Lagrangian multiplier
    tau : float
    r : shifted versions of r
    I : identity matrix
    turns : number of iterations
    
    Returns:
    iter_h : 1d array, with final improved version of initial h
    '''
    iterations = 0
    while iterations < turns:
        new_err = new_h(h, s, lm, tau, r, I)
        h = new_err
        iterations += 1
    return h


'''MCS'''

def mcs(fitsdata, lm, turns, maxiter):
    '''
    Define and perform several iterations of MCS deconvolution.
    
    Parameters:
    fitsdata : data
    lm : Lagrangian multiplier
    turns : number of iterations for improving error distribution h
    maxiter : number of iterations of MCS deconvolution
    
    Returns:
    fitsdata : new version of original data
    '''
    iteration = 0

    while iteration < maxiter:
        if iteration == 0:
            F = deconvolve(fitsdata -  np.median(fitsdata,axis=0) , S.reshape(
                    19, 19), ifftshift=True)
            iterat = 10
            # use Richardson Lucy deconvolution to estimate initial error h
            deconvolved_RL = restoration.richardson_lucy(F, S.reshape(19,19), 
                                                         iterations=iterat)
            h = (F - deconvolved_RL).reshape(361)
            # plot each error distribution
#            plt.imshow(h.reshape(19,19), cmap='gray')
#            plt.title('h')
#            plt.text(0,0,'{}'.format(np.mean(h)))
#            plt.colorbar()
#            plt.show()
            # get initial guesses for intensities of 4 point sources
            a = [] # might be better idea to update a within mcs function
            a.append(np.sum(F.reshape(361)[c[0]-19:c[0]-16]) + np.sum(
                    F.reshape(361)[c[0]-1:c[0]+2]) + np.sum(
                    F.reshape(361)[c[0]+18:c[0]+21]))
            a.append(np.sum(F.reshape(361)[c[1]-19:c[1]-16]) + np.sum(
                    F.reshape(361)[c[1]-1:c[1]+2]) + np.sum(
                    F.reshape(361)[c[1]+18:c[1]+21]))
            a.append(np.sum(F.reshape(361)[c[2]-19:c[2]-16]) + np.sum(
                    F.reshape(361)[c[2]-1:c[2]+2]) + np.sum(
                    F.reshape(361)[c[2]+18:c[2]+21]))
            a.append(np.sum(F.reshape(361)[c[3]-19:c[3]-16]) + np.sum(
                    F.reshape(361)[c[3]-1:c[3]+2]) + np.sum(
                    F.reshape(361)[c[3]+18:c[3]+21]))
            a = np.asarray(a)
            tau = 0 # corresponds to sum over k in summation expression
            for k in range(M):
                tau += (a[k]*r[:, c[k]])
            improved_h = iterate_new_h(h, s, lm, tau, r, I, turns)
            reconvolved_improved_h = reconvolve(improved_h.reshape(19,19), 
                    S.reshape(19,19), ifftshift = True)
            new_image = fitsdata  -reconvolved_improved_h
            fitsdata = new_image      
            iteration += 1
        else:
            F = deconvolve(fitsdata -  np.median(fitsdata,axis=0) , S.reshape(
                    19, 19), ifftshift=True)
            iterat = 9
            deconvolved_RL = restoration.richardson_lucy(F, S.reshape(19,19),
                     iterations=iterat)
            rl_h = (F - deconvolved_RL).reshape(361)
            h =  (improved_h - rl_h)
#            plt.imshow(h.reshape(19,19), cmap='gray')
#            plt.title('h')
#            plt.text(0,0,'{}'.format(np.mean(h)))
#            plt.colorbar()
#            plt.show()
            #initial guesses of intensities for 4 point sources
            a = [] # might be better idea to update a within mcs function
            a.append(np.sum(F.reshape(361)[c[0]-19:c[0]-16]) + np.sum(
                    F.reshape(361)[c[0]-1:c[0]+2]) + np.sum(
                    F.reshape(361)[c[0]+18:c[0]+21]))
            a.append(np.sum(F.reshape(361)[c[1]-19:c[1]-16]) + np.sum(
                    F.reshape(361)[c[1]-1:c[1]+2]) + np.sum(
                    F.reshape(361)[c[1]+18:c[1]+21]))
            a.append(np.sum(F.reshape(361)[c[2]-19:c[2]-16]) + np.sum(
                    F.reshape(361)[c[2]-1:c[2]+2]) + np.sum(
                    F.reshape(361)[c[2]+18:c[2]+21]))
            a.append(np.sum(F.reshape(361)[c[3]-19:c[3]-16]) + np.sum(
                    F.reshape(361)[c[3]-1:c[3]+2]) + np.sum(
                    F.reshape(361)[c[3]+18:c[3]+21]))
            a = np.asarray(a)
            tau = 0 # corresponds to sum over k in summation expression
            for k in range(M):
                tau += (a[k]*r[:, c[k]])
            improved_h = iterate_new_h(h, s, lm, tau, r, I, turns)
            reconvolved_improved_h = reconvolve(improved_h.reshape(19,19), 
                                        S.reshape(19,19), ifftshift = True)
            new_image = fitsdata  - reconvolved_improved_h
            fitsdata = new_image      
            iteration += 1
    return fitsdata

iterated_image = mcs(fitsdata, lm, 4,11)

hs = []
for t in range(1000):
    hs.append(np.random.rand(361)*np.random.randint(-100, 100, 361))
phis = []
for i in range(1000):
    phis.append(phi(hs[i], s, lm, 1, r, I))
    
plt.scatter([np.mean(x) for x in hs], phis)
plt.xlabel('Mean value of error distribution')
plt.ylabel(r'$\phi$')
#plt.savefig('Phi_minimum')
plt.show()
  
'''Generate plots and write output to FITS file'''

# original data
plt.imshow(fitsdata, cmap='gray')
plt.axis('off')
plt.tight_layout()
#plt.savefig('Cropped_original_data')
plt.show()

## total psf T
#plt.imshow(T)
#plt.colorbar()
#plt.title('Total PSF T')
#plt.show()
#
## optimal sampling PSF R
#plt.imshow(R.reshape(19,19))
#plt.colorbar()
#plt.title('Optimally sampled PSF R')
#plt.show()
#
## psf s, convolved with R gives T
#plt.imshow(S.reshape(19,19))
#plt.colorbar()
#plt.title('PSF S')
#plt.show()
#
## deconvolved data F
#plt.imshow(F, cmap='gray')
#plt.colorbar()
#plt.title('F: original data deconvolved with S')
#plt.show()

# mcs iterated image
plt.imshow(iterated_image, cmap='gray')
plt.colorbar()
plt.title('iterate mcs')
plt.show()

# zoom in
mod_iter_image = iterated_image[2:18, 2:17]
mod_iter_image = mod_iter_image/np.max(mod_iter_image)
mod_iter_image[mod_iter_image<0] = 0
plt.imshow(mod_iter_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
#plt.savefig('Cropped_mcs_output')
#hdu = fits.PrimaryHDU(mod_iter_image)
#hdu.writeto('MCS-output.fits')
#plt.imshow(mod_iter_image, cmap='gray')
#plt.colorbar()
#plt.title('cropped iterated mcs')
#plt.show()