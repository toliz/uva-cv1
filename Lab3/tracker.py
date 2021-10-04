import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy.ndimage import gaussian_filter, rotate

from scipy import signal


img_fol = '/content/imgs/lab3-assignment/images/toy'

x_train = np.zeros(((len(os.listdir(img_fol))),240, 320), dtype = np.float32)
imgs_path = []
for i,img_file in enumerate(sorted(os.listdir(img_fol))):
  #print(img_file)
  imgs_path.append(os.path.join(img_fol, img_file))
  img = cv2.imread(os.path.join(img_fol, img_file))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.equalizeHist(img)
  
  x_train[i] = img.astype(np.float32)
  #modify images for improvement
  x_train[i] = cv2.GaussianBlur(x_train[i],(5,5),cv2.BORDER_DEFAULT)
  x_train[i] = cv2.bilateralFilter(x_train[i], 15, 90, 90)
  
  
imgs_path = sorted(imgs_path)


#define functions 
def of_flow(im1, im2, window_size, min_quality=0.01):

    c = 10000
    d = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, c, min_quality, d)

    w = int(window_size/2)

    im1 = im1 / 255
    im2 = im2 / 255

    #Convolve to get gradients w.r.to X, Y and T dimensions
    kx = np.array([[-1, 1], [-1, 1]])
    ky = np.array([[-1, -1], [1, 1]])
    kt = np.array([[1, 1], [1, 1]])

    fx = signal.convolve2d(old_frame,  kx)              #Gradient over X
    fy = signal.convolve2d(old_frame,  ky)              #Gradient over Y
    ft = signal.convolve2d(new_frame,  kt) - signal.convolve2d(old_frame,  kt)  #Gradient over Time


    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature in feature_list:        #   for every corner
            
            j, i = feature.ravel()		#   get cordinates of the corners (i,j). They are stored in the order j, i
            i, j = int(i), int(j)		#   i,j are floats initially

            I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0],1))
            A = np.vstack((I_x, I_y)).T

            U = np.matmul(np.linalg.pinv(A), b)

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)


def draw_quiver2(u,v,beforeImg,i):
    count =i
    scale = 3
    #ax = plt.figure().gca()
    fig, (ax1) = plt.subplots(1, 1)

    ax1.imshow(beforeImg)
    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            ax1.quiver(j,i,dx,dy, color='green')


    plt.draw()
    plt.show()
    plt.axis('off')
    ax1.axis("Off")
    fig.savefig('/content/sample_data/temp.jpg', dpi=300, bbox_inches='tight')
    img_frame = cv2.imread('./temp.jpg')

    
    return img_frame

#start main function 

#send first and second frame for tracking 
img1 = x_train[0]
movie = []
for i in range(0,len(x_train)):
  print(i,i+1)
  U, V = optical_flow( x_train[i], x_train[i+1], window_size=15, min_quality=0.01)
  frame = cv2.imread(imgs_path[i])
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  line_color = (0, 255, 0) #  Green
  res = draw_quiver2(U,V,frame,i)
  movie.append(res)

#use movie frame to convert to movie

movie2 = np.stack(movie)

video_mk("person_toy2.avi", movie2, fps=3)
