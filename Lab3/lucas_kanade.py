import numpy as np 

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from scipy.ndimage import convolve


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def lucas_kanade(img1, img2, w=15):
    # image preprocessing
    if img1.ndim == 3:
        img1 = rgb2gray(img1)
    if img2.ndim == 3:
        img2 = rgb2gray(img2)

    img1 = img1 / img1.max()
    img2 = img2 / img2.max()

    # spatial and time image derivatives
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)

    Ix = convolve(input=img1, weights=np.array([[1,0,-1], [1,0,-1], [1,0,-1]]))
    Iy = convolve(input=img1, weights=np.array([[1,1,1], [0,0,0], [-1,-1,-1]]))
    It = img2 - img1
    
    # Lucas-Kanade algorithm
    wid , hei = img1.shape
    for x in range(w//2, wid - w//2, w):
      for y in range(w//2, hei - w//2, w):

        Ix_local = (Ix[x - w//2 : x + w//2+1, y - w//2 : y + w//2+1]).flatten()
        Iy_local = (Iy[x - w//2 : x + w//2+1, y - w//2 : y + w//2+1]).flatten()
        It_local = (It[x - w//2 : x + w//2+1, y - w//2 : y + w//2+1]).flatten()

        A = np.hstack([Ix_local, Iy_local]).reshape(-1, 2)
        b = -It_local.reshape(-1, 1)

        flow = np.linalg.inv(A.T @ A) @ A.T @ b

        u[x - w//2 : x + w//2+1, y - w//2 : y + w//2+1] = flow[0]
        v[x - w//2 : x + w//2+1, y - w//2 : y + w//2+1] = flow[1]

    return (u, v)


if __name__ == '__main__':
    # Manual setting of scales for quiver
    scale = {'Car': 0.02, 'Coke': 0.05}

    for img in ['Car', 'Coke']:
        img1 = mpimg.imread(f'images/{img}1.jpg')
        img2 = mpimg.imread(f'images/{img}2.jpg')

        u, v = lucas_kanade(img1, img2)

        r = (u**2 + v**2).flatten()
        
        plt.figure(f'Lucas Kanade - {img}')
        plt.imshow(img1, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        
        #hyper paramater, for car used 20 and coke used 15
        a, b = np.meshgrid(
            np.arange(0, img1.shape[1]),
            np.arange(0, img1.shape[0]),
        )

        plt.quiver(
            a[::15, ::15], b[::15, ::15],
            u[::15, ::15], v[::15, ::15],
            color='r',  angles='xy', scale_units='xy', scale=scale[img],
        )

        plt.savefig(f'figures/2/{img}.eps', format='eps', bbox_inches='tight')
