import cv2 
import numpy as np

img = cv2.imread('awb.jpg')

img= img.transpose(2, 0, 1).astype(np.uint32)
ab = np.mean(img[1])
img[0] = np.minimum(img[0] * (ab / np.mean(img[0])), 255)
img[2] = np.minimum(img[2] * (ab / np.mean(img[2])), 255)
img = img.transpose(1, 2, 0).astype(np.uint8)

cv2.imwrite('gray_world_image2.jpg', img)