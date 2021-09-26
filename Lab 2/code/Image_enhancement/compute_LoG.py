from scipy.signal import convolve2d
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def compute_LoG(image, LOG_type):
    img_org = image.copy()
    #create an empty nup image 
    

    def dnorm(x, mu, sd):
        const = 1 / (np.sqrt(2 * np.pi) * sd)
        pow = np.exp(-np.power((x - mu) / sd, 2) / 2)
        return  const * pow 

    def log_kernel(ksize, sig = 1):
        v = np.power(sig, 2)
        x = np.linspace(-(ksize//2), ksize//2, ksize)
        k1 = np.linspace(-(ksize//2), ksize//2, ksize)

        for i in range(ksize):
            k1[i] = dnorm(x[i], 0, sd = sig)
        #compute kernel
        k2d = np.outer(k1.T, k1.T)

        aa,bb = np.meshgrid(x,x)
        term_2 = np.power(aa,2) + (np.power(bb,2))
        term_2 = 2 * v - term_2
        k2d = -k2d * term_2 / np.power(v,2)
        return k2d

    def gauusian_edge(image, ksize, sig):
        
        kernel_2D = log_kernel(ksize, sig=sig)

        log_image = convolve2d(image, kernel_2D)
        edge_img  = cv2.normalize(log_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return edge_img

    if LOG_type == 1:
        h,w, = img_org.shape
        ans = np.zeros((h,w,1))
        #method 1
        #apply gaussian first 
        #image = image/(255.0)
        #gaussian_kernel = cv2.getGaussianKernel(ksize=5,sigma=0.5)
        image = cv2.GaussianBlur(image, ksize=(15,15), sigmaX=0.5,sigmaY=0)
        #computer gaussian 
        image = gauusian_edge(image, ksize=9, sig=1.4)
        #save in one channel of output
        image = image *255
        #cv2.imwrite('laplacian_method1.jpg',image)
        #create a subploit of method 1 and original image
        #store images in ans
        h,w, = image.shape
        ans = np.zeros((h,w,2))
        ans[:,:,0] = image.copy()


    elif LOG_type == 2:
        #print('Not implemented\n')
        image = gauusian_edge(image, ksize=9, sig=1.4)
        #save in one channel of output
        image = image *255
        #cv2.imwrite('laplacian_method2.jpg',image)
        h,w, = image.shape
        ans = np.zeros((h,w,2))
        ans[:,:,0] = image.copy()


    elif LOG_type == 3:
        
        #print('Not implemented\n')
        #use different ksize and sig
        img1 = gauusian_edge(image, ksize=9, sig=1.4)
        #save in one channel of output
        img1 = img1 *255
        #cv2.imwrite('laplacian_method3_img1.jpg',img1)
        #do img2 now 
        img2 = gauusian_edge(image, ksize=9, sig=3.5)
        #save in one channel of output
        img2 = img2 *255
        #cv2.imwrite('laplacian_method3_img2.jpg',img2)
        #store img1,img2
        h,w, = img1.shape
        ans = np.zeros((h,w,2))
        ans[:,:,0] = img1.copy()
        ans[:,:,1] = img2.copy()


    return ans

image = cv2.imread('./images/image1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image_rgb = image_rgb.astype(np.float32)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.astype(np.float32)

ans = compute_LoG(image, 3)

#visualise images and save them 
h,w,c = ans.shape

#ans is either single image or two images 
print(ans.shape)
if(c==1):
    #plt.imshow(input_image, cmap='gray', vmin=0, vmax=255)
    #cv2.imwrite('Output.jpg',ans[:,:,0])
    fig, ax = plt.subplots(1,2, figsize=(15,15))
    a = ans[:,:,0]

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for ax1 in ax:
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax[0].imshow(image_rgb)
    ax[1].imshow(a, cmap='gray', vmin=0, vmax=255)

    ax[0].set_title("Original Image")
    ax[1].set_title("Laplace")

    fig.savefig('./Output_Laplace1.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig('Output_Laplace1.jpg', dpi=300, bbox_inches='tight')

elif (c>1):
    #for gray image visualise function
    a = ans[:,:,0]
    b = ans[:,:,1]
   
    #convert to 255 range
    #input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    fig, ax = plt.subplots(1,3, figsize=(15,15))

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for ax1 in ax:
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax[0].imshow(image_rgb, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(a, cmap='gray', vmin=0, vmax=255)
    ax[2].imshow(b, cmap='gray', vmin=0, vmax=255)

    ax[0].set_title("Original Image")
    ax[1].set_title("Sigma 1 ")
    ax[2].set_title("Sigma 2")

    fig.savefig('./Output_Laplace2.eps', format='eps', dpi=300, bbox_inches='tight')
    fig.savefig('Output_Laplace2.jpg', dpi=300, bbox_inches='tight')