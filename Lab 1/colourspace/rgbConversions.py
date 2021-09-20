import numpy as np
import cv2

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    h,w,c = input_image.shape
    new_image = np.zeros((h,w,4))
    
    gray = np.max(input_image,axis=-1,keepdims=1)/2+np.min(input_image,axis=-1,keepdims=1)/2
    gray = gray[:,:,0]
    #print(gray.shape)
    new_image[:,:,0] = gray



    # average method
    gray = input_image.mean(axis=-1,keepdims=1) 
    gray = gray[:,:,0]
    #print(gray.shape)
    new_image[:,:,1] = gray


    # luminosity method
    #luminosity method
    R,G,B = cv2.split(input_image)
    gray = 0.2126 * R + 0.7152 * G + 0.0722 * B
    #print(gray.shape)
    new_image[:,:,2] = gray


    # built-in opencv function 
    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    #print(gray.shape)
    new_image[:,:,3] = gray
    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    #opponent color space
    #new_image = input_image 
    #take cv2 split 
    R,G,B = cv2.split(input_image)

    rc = (R-G)/np.sqrt(2) 
    gc = (R+G-2*B)/np.sqrt(6)
    bc = (R+G+B)/np.sqrt(3)
    channels = [rc,gc,bc]
    for x in channels:
        #x = (x - x.min)/(x.max-x.min)
        x = (x-np.min(x)) / (np.max(x) - np.min(x))

    #merge image
    new_image = cv2.merge((rc,gc,bc))

    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    norm = np.zeros_like(input_image)
    norm = norm.astype(np.float32)
    norm_rgb = np.zeros_like(input_image)
    norm_rgb = norm_rgb.astype(np.uint8)
    #input_image = cv2.normalize(input_image,  norm, 0, 255, cv2.NORM_MINMAX)
    r,g,b = cv2.split(input_image)
    sum = r+g+b

    norm[:,:,0]=r/sum*255.0
    norm[:,:,1]=g/sum*255.0
    norm[:,:,2]=b/sum*255.0

    new_image=cv2.convertScaleAbs(norm)
    return new_image
