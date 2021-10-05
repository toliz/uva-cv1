import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy.ndimage import gaussian_filter, rotate

from scipy import signal

img_fol = '/content/lab3-assignment/images/doll'

img = os.listdir(img_fol)[0]
img = cv2.imread(os.path.join(img_fol, img))
w,h,_ = img.shape



x_train = np.zeros(((len(os.listdir(img_fol))),w, h), dtype = np.float32)
imgs_path = []
for i,img_file in enumerate(sorted(os.listdir(img_fol))):
  #print(img_file)
  imgs_path.append(os.path.join(img_fol, img_file))
  img = cv2.imread(os.path.join(img_fol, img_file))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #img = cv2.equalizeHist(img)
  
  x_train[i] = img.astype(np.float32)
  #modify images for improvement
  #x_train[i] = cv2.GaussianBlur(x_train[i],(5,5),cv2.BORDER_DEFAULT)
  #x_train[i] = cv2.bilateralFilter(x_train[i], 15, 90, 90)
  
  
imgs_path = sorted(imgs_path)

#define functions 


def calc_mag(u, v):
    ct = 0.0
    inc = 3
    total = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            fy = v[i,j] * inc
            fx = u[i,j] * inc
            ct += 1
            calc = (fx**2 + fy**2)**0.5
            total += calc

    avg = total / ct

    return avg

def draw_quiver2(u,v,frame,i):
    count =i
    scale = 1
    #ax = plt.figure().gca()
    fig, (ax1) = plt.subplots(1, 1)
    ax1.axis("Off")

    canvas = plt.gca().figure.canvas
    canvas.draw()

    ax1.imshow(frame)
    value = calc_mag(u, v)

    for i in range(0, u.shape[0], 1):
        for j in range(0, u.shape[1],1):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            calc = (dx**2 + dy**2)**0.5
            if calc > (value):
              ax1.quiver(j,i,dx,dy,  angles='xy', scale_units='xy', scale=0.1, color='green')


    plt.draw()
    plt.show()
    plt.axis('off')
    ax1.axis("Off")
    fig.savefig('./temp.jpg', dpi=300, bbox_inches='tight')
    img_frame = cv2.imread('./temp.jpg')
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
    return img_frame



#define functions 
def of_flow(img1, img2, w, min_quality=0.01):

    
    img1 = img1 / 255
    img2 = img2 / 255
    _, r, c = harris_corner_detector(img1)
    c1 = np.vstack((r, c))

    w = int(w/2)


    k1 = np.array([[-1, 1], [-1, 1]])
    k2 = np.array([[-1, -1], [1, 1]])
    k3 = np.array([[1, 1], [1, 1]])


    u = np.zeros_like(img1)
    v = u

    dx = signal.convolve2d(img1,  k1)             
    dy = signal.convolve2d(img1,  k2)             
    dt = signal.convolve2d(img2,  k3) - signal.convolve2d(img1,  k3)

    #iterate only c1s
    for i in range(corners.shape[1]):
            j, i = corners[0, i], corners[1, i] 
            i, j = int(i), int(j)   

            d2x = dx[i-w:i+w+1, j-w:j+w+1].flatten()
            d2y = dy[i-w:i+w+1, j-w:j+w+1].flatten()
            d2t = dt[i-w:i+w+1, j-w:j+w+1].flatten()

            B = np.reshape(d2t, (d2t.shape[0],1))
            A = np.vstack((d2x, d2y)).T

            U = np.matmul(np.linalg.pinv(A), B)

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)




#send first and second frame for tracking 
img1 = x_train[0]
movie = []
for i in range(0,(len(x_train)-2)):
  print(i,i+1)
  # modify images for of and blur 
  U, V = of_flow( x_train[i], x_train[i+1], w=15, min_quality=0.01)
  #got u, v for first set of image 0, image 1 
  #now add it to a numpy array and plot it 
  frame = cv2.imread(imgs_path[i])
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  line_color = (0, 255, 0) #  Green
  res = draw_quiver2(U,V,frame,i)

  movie.append(res)

#use movie list to obtain movie 

movie2 = np.stack(movie)

def _save_video(filename, array, fps=10):
    f, height, width, c = array.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for i in range(f):
        out.write(array[i, :, :, ::-1])

_save_video("of_flow_output_video.avi", movie2, fps=3)


