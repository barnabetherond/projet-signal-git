import cv2
import numpy as np
import tqdm


# ------------------- Loading / Saving video -----------------------------------

def load_video(video_path):

    """
    Convert the video into a stack of images
    
    Parameters
    ----------
    
    video_path: string
      location of the video
      
    Return
    ------
    
    image_sequence: array-like
      video as a stack of images
      
    fps: int
      number of frames per second in the video
    """
    
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
        image_sequence.append(frame[:, :, ::-1])

    video.release()
    return np.asarray(image_sequence), fps


def save_video(video, saving_path, fps):

    """
    Save a video
    
    Parameters
    ----------
    
    video: array-like
      stack of images constituting the video
      
    saving_path: string
      location where to save the video
      
    fps: int
      number of frames per second 
    """
    (height, width) = video[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video)), ascii=True, desc="Saving Video"):
        writer.write(video[i][:, :, ::-1])

    writer.release()


# ------------------- Color conversion -----------------------------------

def rgb2yiq(rgb_image):

    """
    Converts a RGB image into a YIQ image 
    
    Parameters
    ----------

    rgb_image: array-like
      image in YIQ color space
          
    Return
    ------
    
    image: array-like
      image in YIQ color space
    """
    
    yiq_from_rgb = (
    np.array(
            [
                [0.29900000,  0.58700000,  0.11400000],
                [0.59590059, -0.27455667, -0.32134392],
                [0.21153661, -0.52273617,  0.31119955]
            ]
        )
    ).astype(np.float32)
    
    image = rgb_image.astype(np.float32)
    return image @ yiq_from_rgb.T


def yiq2rgb(yiq_image):

    """
    Converts a YIQ image into a RGB image 
    
    Parameters
    ----------

    yiq_image: array-like
      image in YIQ color space
          
    Return
    ------
    
    image: array-like
      image in RGB color space
    """

    yiq_from_rgb = (
    np.array(
            [
                [0.29900000,  0.58700000,  0.11400000],
                [0.59590059, -0.27455667, -0.32134392],
                [0.21153661, -0.52273617,  0.31119955]
            ]
        )
    ).astype(np.float32)
    
    rgb_from_yiq = np.linalg.inv(yiq_from_rgb)
    image = yiq_image.astype(np.float32)
    return image @ rgb_from_yiq.T


# ---------------------- Gaussian kernel ---------------------------------------

gaussian_kernel = (
    np.array(
        [
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ]
    )
    / 256
)

