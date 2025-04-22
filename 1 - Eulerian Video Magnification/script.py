from algorithms import *              
from utils import *


def evm(images, fps, level, alpha, freq_range):

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
    # A CODER. utiliser le noyau gaussien défini dans utils.py
    x, y, t, p = images.shape
    pyramid = np.zeros_like(images)
    for ith_frame in range(t):
        pyramid[:,:,ith_frame] = generateGaussianPyramid(images[:,:,ith_frame], gaussian_kernel, level)
        
                    
    # Filter the pyramid  
    # A CODER 
    filtered_images = alpha * apply_temporal_filter(pyramid, fps, freq_range)
        

    # Video reconstruction   
    output_video = np.zeros_like(images)
    for i in tqdm.tqdm(range(images.shape[0]), ascii=True, 
     desc="Video reconstruction"):
    
        reconstructed_image = rgb2yiq(images[i]) + filtered_images[i]
        reconstructed_image = yiq2rgb(reconstructed_image)
        output_video[i] = np.clip(reconstructed_image, 0, 255)
             
    return output_video



if __name__ == "__main__":
   
    # Load video
    video_path = './video/face.mp4'
    images, fps = load_video(video_path=video_path)
    
    # Parameters
    level = 6
    alpha = 50
    low_omega = 0 # A MODIFIER
    high_omega = 1e6 # A MODIFIER
    freq_range = (low_omega, high_omega)
    #filtered_image = apply_temporal_filter(images, fps, freq_range)
    # Motion amplification
    #processed_video = evm(images, fps, level, alpha, freq_range)
    processed_video = gaussian_evm(images, fps, level, alpha, freq_range)
    
    # Save video
    saving_path = './results/processed.mp4'
    save_video(video=processed_video, saving_path=saving_path, fps=fps)


# +
video_path = './video/face.mp4'
images, fps = load_video(video_path=video_path)

apply_temporal_filter(images, fps, freq_range)
print(images.shape)
# -


if __name__ == "__main__":
   
    # Load video
    video_path = './video/face.mp4'
    images, fps = load_video(video_path=video_path)
    image = images[:, :, 0]

    saving_path = '/results/gaussienne_downsample.jpg'
    save_image(image=downsample(image, gaussian_kernel), saving_path=saving_path)



