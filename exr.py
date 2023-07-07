import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_exr(fpath):
    im = cv2.imread(fpath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0]
    # im = im/im.max()
    return im

if __name__=='__main__':
    plt.imshow(read_exr("/bigpen/simulator_data/LEC_fast/LEC_depth00136.exr"))
    plt.show()