def compute_gradient(image):
    print('Not implemented\n')
    return Gx, Gy, im_magnitude,im_direction
import matplotlib.pyplot as plt


import numpy as np
from scipy.signal import convolve2d
import cv2 
# Load sample data
img = cv2.imread('./images/image1.jpg')
image = img

#define horizontal and Vertical sobel kernels
Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
Gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# #define kernal convolution function
# # with image X and filter F
# def convolve(X, F):
#     # height and width of the image
#     X_height = X.shape[0]
#     X_width = X.shape[1]
    
#     # height and width of the filter
#     F_height = F.shape[0]
#     F_width = F.shape[1]
    
#     H = (F_height - 1) // 2
#     W = (F_width - 1) // 2
    
#     #output numpy matrix with height and width
#     out = np.zeros((X_height, X_width))
#     #iterate over all the pixel of image X
#     for i in np.arange(H, X_height-H):
#         for j in np.arange(W, X_width-W):
#             sum = 0
#             #iterate over the filter
#             for k in np.arange(-H, H+1):
#                 for l in np.arange(-W, W+1):
#                     #get the corresponding value from image and filter
#                     a = X[i+k, j+l]
#                     w = F[H+k, W+l]
#                     sum += (w * a)
#             out[i,j] = sum
#     #return convolution  
#     return out


#normalizing the vectors
sob_x = convolve2d(image, Gx) / 8.0
sob_y = convolve2d(image, Gy) / 8.0

cv2.imwrite('Sobx.jpg', sob_x)
cv2.imwrite('Soby.jpg', sob_y)

#calculate the gradient magnitude of vectors
sob_out = np.sqrt(np.power(sob_x, 2) + np.power(sob_y, 2))
# mapping values from 0 to 255
sob_out = (sob_out / np.max(sob_out)) * 255

sob_direct = np.arctan2(sob_y, sob_x) * (180 / np.pi) % 180

#output images
cv2.imwrite('sobel_mag.jpg', sob_out)
cv2.imwrite('sobel_direction.jpg', sob_direct)


#display function 

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
