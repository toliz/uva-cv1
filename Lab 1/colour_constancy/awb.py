import cv2 
import numpy as np

img = cv2.imread('awb.jpg')



def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0] * (mu_g / np.average(nimg[0])), 255)
    nimg[2] = np.minimum(nimg[2] * (mu_g / np.average(nimg[2])), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

gray = grey_world(img)
#transponse using library now 
#gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
cv2.imwrite('gray_world_image.jpg', gray)