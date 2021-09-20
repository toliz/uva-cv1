import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    #using matplotlib instead of opencv since the opencv GUI sometimes causes a glitch
    #split the image into channels and visualise it and store it in a folder results with corresponding filename 
    #already output image is got here so split channels and visualise
    #split image into 3 channels 
    # visualising code 
    #subplot(r,c) provide the no. of rows and columns
    #print(len(input_image))
    #input_image = input_image*255
    #input_image = input_image.astype(np.uint8)
    input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    h,w,c = input_image.shape

    #channels = input_image.shape[3]
    if(c<3):
        plt.imshow(input_image, cmap='gray', vmin=0, vmax=255)
        cv2.imwrite('Output.jpg', input_image)
    elif (c>3):
        #for gray image visualise function
        a = input_image[:,:,0]
        b = input_image[:,:,1]
        c = input_image[:,:,2]
        d = input_image[:,:,3]

        #convert to 255 range
        #input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        fig, ax = plt.subplots(1,4, figsize=(15,15))

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for ax1 in ax:
            ax1.set_xticks([])
            ax1.set_yticks([])

        ax[0].imshow(a, cmap='gray', vmin=0, vmax=255)
        ax[1].imshow(b, cmap='gray', vmin=0, vmax=255)
        ax[2].imshow(c, cmap='gray', vmin=0, vmax=255)
        ax[3].imshow(d, cmap='gray', vmin=0, vmax=255)

        ax[0].set_title("Lightness method")
        ax[1].set_title("Average method")
        ax[2].set_title("Luminosity")
        ax[3].set_title("Built in OpenCV method")

        fig.savefig('./Output.jpg')

    else:

        fig, ax = plt.subplots(1,4, figsize=(15,15))

        r,g,b = cv2.split(input_image)

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        for ax1 in ax:
            ax1.set_xticks([])
            ax1.set_yticks([])

        ax[0].imshow(input_image)
        ax[1].imshow(r,  cmap='gray', vmin=0, vmax=255)
        ax[2].imshow(g,  cmap='gray', vmin=0, vmax=255)
        ax[3].imshow(b,  cmap='gray', vmin=0, vmax=255)

        ax[0].set_title("Output")
        ax[1].set_title("Channel-1")
        ax[2].set_title("Channel-2")
        ax[3].set_title("Channel-3")

        fig.savefig('./Output1.jpg')
        #cv2.imwrite('Output2.jpg', input_image)



