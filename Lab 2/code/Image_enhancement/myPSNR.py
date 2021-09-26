import cv2
import numpy as np


def myPSNR(orig_image, approx_image):
    assert orig_image.shape == approx_image.shape, "please use images of the same shape"

    # convert pixels to float values
    if orig_image.dtype != np.dtype('float32'):
        orig_image = orig_image.astype('float32')

    if approx_image.dtype != np.dtype('float32'):
        approx_image = approx_image.astype('float32')

    # compute RMSE
    RMSE = np.sqrt(np.mean( (orig_image - approx_image) ** 2 ))
    PSNR = 20 * np.log10(orig_image.max() / RMSE)
    
    return PSNR


if __name__ == '__main__':
    orig_image = cv2.imread('./images/image1.jpg')
    approx_image = cv2.imread('./images/image1_gaussian.jpg')

    print(f'PSNR = {myPSNR(orig_image, approx_image)} dB.')
