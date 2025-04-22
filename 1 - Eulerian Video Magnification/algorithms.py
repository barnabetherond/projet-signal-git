import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
from utils import *

# ------------------ Upsampling/Downsampling -----------------------------------

# +
def application_noyau(part, kernel):
    n = len(part)
    somme = 0
    for u in range(n):
        for k in range(n):
            somme += part[u, k]*kernel[u, k]
    return somme
            
def apply_filter(image, kernel):
    x, y, _ = image.shape
    N = len(kernel)

    diago = int(N/2) 

    image_filtre = np.zeros_like(image)

    for i in range(x):
        for j in range(y):
            part = np.zeros((N, N, 3))
            if i == 0 and j == 0:
                part[diago:N, diago:N] = image[diago:N, diago:N]
            elif i == 0 and j == y-diago:
                part[diago:N, 0:diago] = image[diago:N, 0:diago]
            elif i == x-diago and j == 0:
                part[0:diago, diago:N] = image[0:diago, N:diago]
            elif i == x-diago and j == y-diago:
                part[0:diago, 0:diago] = image[0:diago, 0:diago]
            elif i == 0:
                part[diago:N, :] = image[diago:N, :]
            elif i == x-diago:
                part[0:diago, :] = image[0:diago, :]
            elif j == 0:
                part[:, diago:N] = image[:, diago:N]
            elif j == y-diago:
                part[:, 0:diago] = image[:, 0:diago]
            else:
                part = image[i-diago:i+diago, j-diago:j+diago, :]
            
            image_filtre[i, j] = application_noyau(part, kernel)
    return image_filtre
    
def downsample(image, kernel):

    """
    Downsample the image 
    
    A Gaussian filter is applied to the image, which is then downsampled by
    a factor 2
    
    Parameters
    ----------
    
    image: array-like
      original image
    kernel: array-like
      kernel of the filter
      
    Return
    ------
    
    out: array-like
      dowsampled image
    """
    x, y, _ = image.shape
    N = len(kernel)
    

    #On applique le filtre gaussien
    image_filtre = apply_filter(image, kernel)

    #On réduit la taille de l'image en sous-échantillonant
    taille_downsample = int(x/2)
    image_downsample = np.zeros((taille_downsample, taille_downsample))
    for i in range (taille_downsample):
        for j in range(taille_downsample):
            image_down_sample[i, j] = image_filtre[2*i, 2*j]
        
    return image_downsample
                    
    
    
# -
def upsample(image, kernel, dst_shape=None):

    """
    Upsample the image up to the specified size
    
    The image is upsampled by first inserting zeros between adjacent pixels. 
    The resulting image is then filtered.
    
    Parameters
    ----------
    
    image: array-like
      original image
      
    kernel: array-like
      kernel of the filter
      
    dst_shape: tuple of ints
      shape of the output image
      
    Return
    ------
    
    out: array-like
      output image
    """
    #On upsample d'abord l'image
    N = len(image)
    taille_upsample = int(2*N)
    image_upsample = np.zeros_like(image)
    for i in range(taille_upsample):
        for j in range(taille_upsample):
            if i%2 == 0 and j%2 == 0:
                image_upsample[i, j] = image[int(i/2), int(j/2)]  
    

    #On applique maintenant le filtre gaussien
    image_finale = apply_filter(image_upsample, kernel)
    return image_finale
    
    # A CODER


# ------------------ Gaussian pyramid -----------------------------------------


def generateGaussianPyramid(image, kernel, level):

    """
    Image filtering using a Gaussian pyramid
    
    Parameters
    ----------
    
    image: array-like
      image to filter
    kernel: array-like
      convolution kernel
    level: int
      number of approximation levels in the pyramid
      
    Return
    ------
    
    output_image: array-like
      filtered image
    """
    for i in range (level):
        image = downsample(image, kernel)
    return image

    # A CODER

# ------------------------- Temporal filter -----------------------------------



def apply_temporal_filter(images, fps, freq_range):

    """
    Apply an ideal temporal bandpass filter to the video
    
    Parameters
    ----------
    
    images: array-like
      stack of images constituting the video
      
    fps: int
      number of frames per second
      
    freq_range: tuple of float
      frequency range for the bandpass filter
      
    Return
    ------
    
    out: array-like
      filtered video represented as a stack of images
    """
    video_filtre = np.zeros_like(images)
    N = len(images[0, 0, :])
    x, y, t, p = images.shape
    f_min, f_max = freq_range
    
    h_fourier = np.zeros_like(images[0, 0, :])
    abcisse_frequentiel = np.fft.fftfreq(N, fps)
    for i in range(N):
        if f_min < abcisse_frequentiel[i] < f_max:
            h_fourier[i] = 1

    
    for pix_x in range(x):
        for pix_y in range(y):
            signal = images[pix_x, pix_y, :]
            fft_signal = np.fft.fft(signal)
            fft_signal_filtré = fft_signal*h_fourier
            video_filtre[pix_x, pix_y, :] = np.fft.ifft(fft_signal_filtré)
            
    return video_filtre
            
    

    # A CODER



