# def compute_gradient(image):
#     print('Not implemented\n')
#     return Gx, Gy, im_magnitude,im_direction
# import matplotlib.pyplot as plt


import numpy as np
from scipy.signal import convolve2d
import cv2 
# Load sample data
img = cv2.imread('./images/image1.jpg')
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = img

#call function here 
def compute_gradient(image):

    #define horizontal and Vertical sobel kernels
    Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    #normalizing the vectors
    sob_x = convolve2d(image, Gx) / 8.0
    sob_y = convolve2d(image, Gy) / 8.0

    cv2.imwrite('Sobx.jpg', sob_x)
    cv2.imwrite('Soby.jpg', sob_y)

    sob_out = np.sqrt(np.power(sob_x, 2) + np.power(sob_y, 2))
    sob_out = (sob_out / np.max(sob_out)) * 255
    sob_direct = np.arctan2(sob_y, sob_x) * (180 / np.pi) % 180

    #output images
    cv2.imwrite('sobel_mag.jpg', sob_out)
    cv2.imwrite('sobel_direction.jpg', sob_direct)

return Gx, Gy, sob_out, sob_direct
#display function 

#function call 
Gx, Gy, sob_out, sob_direct =  compute_gradient(image)

#compute again 
sob_x = convolve2d(image, Gx) / 8.0
sob_y = convolve2d(image, Gy) / 8.0

sob_out = np.sqrt(np.power(sob_x, 2) + np.power(sob_y, 2))
sob_out = (sob_out / np.max(sob_out)) * 255
sob_direct = np.arctan2(sob_y, sob_x) * (180 / np.pi) % 180

fig, ax = plt.subplots(1,5, figsize=(15,15))

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
for ax1 in ax:
    ax1.set_xticks([])
    ax1.set_yticks([])

ax[0].imshow(image_rgb, cmap='gray', vmin=0, vmax=255)
ax[1].imshow(sob_x, cmap='jet', vmin=0, vmax=255)
ax[2].imshow(sob_y, cmap='jet', vmin=0, vmax=255)
ax[3].imshow(sob_out, cmap='jet', vmin=0, vmax=255)
ax[4].imshow(sob_direct, cmap='jet', vmin=0, vmax=255)

ax[0].set_title("Input Image")
ax[1].set_title("Sobel X")
ax[2].set_title("Sobel Y")
ax[3].set_title("Sobel Magnitude")
ax[4].set_title("Sobel Direction")


fig.savefig('./Output_sobel.eps', format='eps', dpi=300, bbox_inches='tight')
fig.savefig('Output_sobel.jpg', dpi=300, bbox_inches='tight')
