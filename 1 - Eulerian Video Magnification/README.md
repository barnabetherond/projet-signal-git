README.md
========================================

Eulerian video magnification code

Data Organization
-----------------
|
|--video/        *# folder containing the video to process*
|--results/      *# experimental images*


Scripts
-----------------

Training set generation
|--script.py         *# Scripts performing Eulerian video magnification*
|--utils.py          *# Methods used to load/save the videos, convert a video in a stack of images, etc. *
|--algorithms.py     *# Library of algorithms used for eulerian video magnification


Requirements
---------------------------
The code relies upon the following Python libraries:
- numpy
- cv2
- tqdm

To install these libraries:

~$ pip install tqdm
~$ pip install opencv-python
