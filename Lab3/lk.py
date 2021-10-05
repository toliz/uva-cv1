import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img1 = cv2.imread('./Coke1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./Coke2.jpg', cv2.IMREAD_GRAYSCALE)

#convert to float 
img1 = img1.astype(np.float32)
img1 = img1.astype(np.float32)


def of_flow_new(img1, img2,w):
    img1 = img1/ 255
    img2 = img2 / 255
        #kernels
    k1 = np.array([[-1,1],[-1, 1]]) / 4
    k2 = np.array([[-1, -1], [1, 1]]) / 4
    k3 = np.array([[1, 1], [1, 1]]) / 4

    dx = ndimage.convolve(input=img1, weights=k1)+ndimage.convolve(input=img2, weights=k1)
    dy = ndimage.convolve(input=img1, weights=k2)+ndimage.convolve(input=img2, weights=k2)
    dt = ndimage.convolve(input=img2, weights=k3)+ndimage.convolve(input=img1, weights=-1*k3)
    u=np.zeros((img1.shape))
    v=np.zeros((img1.shape))
    wid , hei = img1.shape
    for x in range(w//2,wid-w//2):
      for y in range(w//2,hei-w//2):

        dx2 = (dx[x - w // 2:x + w //2 + 1,  y - w // 2:y + w // 2 + 1]).flatten()
        dy2 = (dy[x - w // 2:x + w // 2 + 1, y - w // 2:y + w // 2 + 1]).flatten()
        dt2 = (dt[x - w // 2:x + w // 2 + 1, y - w // 2:y + w // 2 + 1]).flatten()

        B=-1*np.asarray(dt2)
        A=np.asarray([dx2,dy2]).reshape(-1,2)

        flow=np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)),np.transpose(A)),B)

        u[x,y]=flow[0]
        v[x,y]=flow[1]

    return (u, v)

#call to function    
u, v = of_flow_new(img1, img2, 15)





plt.figure()
plt.imshow(img1, cmap='gray', interpolation='bicubic')
plt.title('Lucas Kanade')
plt.axis('off')
#hyper paramater, for car used 20 and coke used 15
th = 15
a = np.arange(0, img1.shape[1], 1)
b = np.arange(0, img1.shape[0], 1)
a, b = np.meshgrid(a, b)

plt.quiver(a[::th, ::th], b[::th, ::th], u[::th, ::th], v[::th, ::th],color='g',  angles='xy', scale_units='xy', scale=0.1)
plt.savefig('./lk_flow.jpg', dpi=300, bbox_inches='tight')
plt.savefig('./lk_flow.eps', format='eps', bbox_inches='tight')
