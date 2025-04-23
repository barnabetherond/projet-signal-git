from algorithms import *              
from utils import *


def evm(images, fps, alpha, freq_range):

    """
    "Naive" Eulerian video magnification
    
    Parameters
    ----------
    
    images: nd.array
      images constituting the video sequence. The i-th frame of the video
      sequence is images[:, :, i]
      
    fps: int
      number of frames per second
      
    alpha: float
      motion amplification factor 
      
    freq_range: tuple (fmin, fmax)
      range of temporal frequencies for which the motion has to be amplified
      
    Return
    ------
    
    output_video: nd.array
      video with amplified motion
      
    """
                   
    # Filtering
    print('Filter pyramid...')                 
    filtered_images = alpha * apply_temporal_filter(images, fps, 
      freq_range=freq_range)

    # Video reconstruction
    print('Video reconstruction...')     
    output_video = np.zeros_like(images)
    for i in range(filtered_images.shape[0]):
    
        reconstructed_image = rgb2yiq(images[i]) + filtered_images[i]
        reconstructed_image = yiq2rgb(reconstructed_image)
        output_video[i] = np.clip(reconstructed_image, 0, 255)
             
    return output_video


def gaussian_evm(images, fps, level, alpha, freq_range):


    """
    Eulerian video magnification based on gaussian pyramids
    
    Parameters
    ----------
    
    images: nd.array
      images constituting the video sequence. The i-th frame of the video
      sequence is images[:, :, i]
      
    fps: int
      number of frames per second
      
    level: int
      number of decomposition levels for the pyramid
      
    alpha: float
      motion amplification factor 
      
    freq_range: tuple (fmin, fmax)
      range of temporal frequencies for which the motion has to be amplified
      
    Return
    ------
    
    output_video: nd.array
      video with amplified motion
      
    """
    
    # Gaussian pyramid
    
    x, y, t, p = images.shape
    pyramid = []
    for ith_frame in range(t):
        pyramid.append(generateGaussianPyramid(images[:,:,ith_frame, :], gaussian_kernel, level))  #Pour chaque frame de la video, on calcule son image de pyramide de Gausse basse résolution
    pyramid = np.array(pyramid)
    #Au vu de comment on a procédé avec la liste pyramid, ici le "temps" est la première dimension, on modifie donc cela pour qu'on se place dans le cadre du sujet     
    pyramid = np.transpose(pyramid, (1, 2, 0, 3))
    
                    
    # Filter the pyramid  
    #On filtre maintenant notre "video basse résolution" à l'aide de la fonction de la q1
    filtered_images_basse_resolution = alpha * apply_temporal_filter(pyramid, fps, freq_range)

    #On remonte maintenant la pyramide

    
    filtered_images = []
    for ith_frame in range(t):
        u = filtered_images_basse_resolution[:, :, ith_frame, :]
        for i in range(level):
            u = upsample(u, gaussian_kernel)
        filtered_images.append(u)
    
    filtered_images = np.array(filtered_images)
    filtered_images = np.transpose(filtered_images, (1, 2, 0, 3))
        

    # Video reconstruction   
    output_video = np.zeros_like(images)
    for i in tqdm.tqdm(range(images.shape[0]), ascii=True, 
     desc="Video reconstruction"):
    
        reconstructed_image = rgb2yiq(images[i]) + filtered_images[i]
        reconstructed_image = yiq2rgb(reconstructed_image)
        output_video[i] = np.clip(reconstructed_image, 0, 255)
             
    return output_video
