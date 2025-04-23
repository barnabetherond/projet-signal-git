import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
from utils import *

# ------------------ Upsampling/Downsampling -----------------------------------

# +

def apply_filter(image):
    x, y, _ = image.shape
    
    imint = np.zeros((x+4,y+4,_))
    imint[2:x+2,2:y+2,:]=image[:,:,:]
    image_filtre=np.zeros(imint.shape, dtype=np.float64)
    image_filtre[2:x+2,2:y+2,:]=(1/256)*(imint[0:x,0:y,:]+imint[4:x+4,4:y+4,:]+imint[0:x,4:y+4,:]+imint[4:x+4,0:y,:]+
                                         4*(imint[0:x,1:y+1,:]+imint[0:x,3:y+3,:]+imint[1:x+1,0:y,:]+imint[1:x+1,4:y+4,:]+imint[3:x+3,0:y,:]+imint[3:x+3,4:y+4,:]+imint[4:x+4,1:y+1,:]+imint[4:x+4,3:y+3,:])+
                                         6*(imint[0:x,2:y+2,:]+imint[4:x+4,2:y+2,:]+imint[2:x+2,0:y,:]+imint[2:x+2,4:y+4,:])+
                                         16*(imint[1:x+1,1:y+1,:]+imint[1:x+1,3:y+3,:]+imint[3:x+3,1:y+1,:]+imint[3:x+3,3:y+3,:])+
                                         24*(imint[1:x+1,2:y+2,:]+imint[3:x+3,2:y+2,:]+imint[2:x+2,1:y+1,:]+imint[2:x+2,3:y+3,:])+
                                         36*imint[2:x+2,2:y+2,:])
    imagef=np.zeros(image.shape, dtype=np.float64)
    imagef[:,:,:]=image_filtre[2:x+2,2:y+2,:]
    return imagef


def downsample(image, kernel):

    
    x, y, _ = image.shape
    
    

    #On applique le filtre gaussien
    image_filtre = apply_filter(image)
    

    #On réduit la taille de l'image en sous-échantillonant

    #on dissocie le cas d'une taille impaire en rajoutant une ligne ou colonne de zéro
    if x%2 == 1:
        image = np.concatenate([image, np.zeros((1, y, 3))])
    elif y%2 == 1:
        for i in range(x):
            image[i] = np.concatenate([image[i], np.array([0, 0, 0])])


    #on crée un tableau de zéros de tailles 2 fois plus petites, puis on ajoute une valeure sur 2

    x_new, y_new, _ = image.shape
    taille_x, taille_y = int(x_new/2), int(y_new/2)
    image_downsample = np.zeros((taille_x, taille_y, 3))
    for i in range (taille_x):
        for j in range(taille_y):
            image_downsample[i, j] = image_filtre[2*i, 2*j]


    #pour éviter les problèmes de tailles impaires etc, l'idéal aurait été d'avoir une liste vide et d'append un élément sur 2
    return image_downsample
                    
    
    
# +
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
    x, y, _ = image.shape
    x_up, y_up = int(2*x), int(2*y)
    image_upsample = np.zeros((x_up, y_up, 3))
    for i in range(x_up):
        for j in range(y_up):
            if i%2 == 0 and j%2 == 0:
                image_upsample[i, j] = image[int(i/2), int(j/2)]  
    

    #On applique maintenant le filtre gaussien
    image_finale = apply_filter(image_upsample)
    return image_finale
    
    
# -

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


# ------------------------- Temporal filter -----------------------------------

# +
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
            
    

    
# -



