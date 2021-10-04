import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img1 = cv2.imread('./images/Coke1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/Coke2.jpg', cv2.IMREAD_GRAYSCALE)

#convert to float 
img1 = img1.astype(np.float32)
img1 = img1.astype(np.float32)



def of_flow(im1,im2,w = 15):
  #derivatives 
  kx = 0.25* np.array(([-1,1],[-1,1]))
  ky = 0.25* np.array(([-1,1],[-1,1]))
  kt = np.fliplr(kx)
  w = int(w)
  fx = cv2.filter2D(im1, -1, kx) + cv2.filter2D(im2, -1, kx)
  fy = cv2.filter2D(im1, -1, ky) + cv2.filter2D(im2, -1, k2)
  ft = cv2.filter2D(im1, -1, -0.25 * np.ones((2,2))) + cv2.filter2D(im2, -1, -0.25 * np.ones((2,2)))

  term = cv2.filter2D(fx **2, -1, w) * cv2.filter2D(fy**2, -1, w) - cv2.filter2D((fx*fy), -1, w)**2
  
  term2[term2==0] = np.inf


  u = (- cv2.filter2D(fy**2, -1, w) * cv2.filter2D(fx*ft, -1, w) + cv2.filter2D(fx*fy, -1, w)*cv2.filter2D(fy*ft, -1, window))/ term
       
  v = cv2.filter2D(fx*ft, -1, w) * cv2.filter2D(fx*fy, -1, window) - cv2.filter2D(fx**2, -1, w) * cv2.filter2D(fy*ft, -1, window) / term

  return u,v

u, v = of_flow(img1, img2, w=15)


#display function 
a = np.arange(0, img1.shape[1], 1)
b = np.arange(0, img1.shape[0], 1)
a,b = np.meshgrid(a,b)


plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Lucas Kanade')
plt.axis('off')
#var
inc = 3
plt.quiver(a[::inc, ::inc], b[::inc, ::inc], u[::inc, ::inc], v[::ince, ::inc], color='green')
plt.savefig('/content/lk_coke.jpg', dpi=300, bbox_inches='tight')
plt.savefig('/content/lk_coke.eps', format='eps', bbox_inches='tight')


