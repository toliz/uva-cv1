import cv2
import numpy as np

from matplotlib import pyplot as plt
from myPSNR import myPSNR


def denoise(image, kernel_type, **kwargs):
    if kernel_type == 'box':
        assert 'kernel_size' in kwargs

        ksize = kwargs['kernel_size']
        if isinstance(ksize, int):
            ksize = (ksize, ksize)

        del kwargs['kernel_size']
        if 'sigma' in kwargs: del kwargs['sigma']

        imOut = cv2.blur(image, ksize, **kwargs)
    
    elif kernel_type == 'median':
        assert 'kernel_size' in kwargs

        ksize = kwargs['kernel_size']
        if isinstance(ksize, tuple):
            ksize = ksize[0]

        del kwargs['kernel_size']
        if 'sigma' in kwargs: del kwargs['sigma']

        imOut = cv2.medianBlur(image, ksize, **kwargs)
    
    elif kernel_type == 'gaussian':
        assert all(arg in kwargs for arg in ['kernel_size', 'sigma'])
        
        ksize = kwargs['kernel_size']
        sigmaX, sigmaY = kwargs['sigma']

        if isinstance(ksize, int):
            ksize = (ksize, ksize)

        del kwargs['kernel_size'], kwargs['sigma']

        imOut = cv2.GaussianBlur(image, ksize, sigmaX, sigmaY, **kwargs)
    
    else:
        print('Operation not implemented')
    
    return imOut


if __name__ == '__main__':
    psnr = np.ndarray([2, 3, 3])
    
    for i, noise in enumerate(['saltpepper', 'gaussian']):
        # read original and poluted (with noise) images
        original_image = cv2.imread(f'images/image1.jpg')
        poluted_image = cv2.imread(f'images/image1_{noise}.jpg')
        
        # report PSNR between original and poluted image.
        print()
        print('Original - Noised PSNR: %.2f dB.' % myPSNR(original_image, poluted_image))
        print('=' * 34)

        # iterate over 6 different filters for denoising
        for j, kernel_type in enumerate(['box', 'median']):
            fig, ax = plt.subplots(1, 4)
            for axis in ax: axis.set_axis_off()

            for k, kernel_size in enumerate([3, 5, 7]):
                denoised_image = denoise(
                    poluted_image,
                    kernel_type=kernel_type,
                    kernel_size=kernel_size,
                )

                # report PSNR
                psnr[i, j, k] = myPSNR(original_image, denoised_image)
                print(f'{noise:^10s} - {kernel_type:^8s} - {kernel_size}: {psnr[i, j, k]:.2f} dB.')

                # plot denoised image
                ax[k+1].imshow(denoised_image)
                ax[k+1].set_title(f'kernel size = {kernel_size}x{kernel_size}', fontsize=5)

            # plot original image next to denoised & save results
            ax[0].imshow(original_image)
            ax[0].set_title('Original', fontsize=5)

            fig.savefig(
                f'../../result_images/{noise}_{kernel_type}.eps',
                pad_inches=0,
                bbox_inches='tight',
            )

    ###############################################################################################
    #                                             PART 2                                          #
    ###############################################################################################
    print()
    print('Gaussian filtering')
    print('=' * 34)
    
    for i, noise in enumerate(['saltpepper', 'gaussian']):
        original_image = cv2.imread(f'images/image1.jpg')
        poluted_image = cv2.imread(f'images/image1_{noise}.jpg')

        fig, ax = plt.subplots(1, 4)
        for axis in ax: axis.set_axis_off()

        # iterate over possible parameters
        for k, sigma in enumerate([1, 2, 3]):
            denoised_image = denoise(
                poluted_image,
                kernel_type='gaussian',
                kernel_size=4*sigma+1,
                sigma=(sigma, sigma),
            )

            # report PSNR
            psnr[i, 2, k] = myPSNR(original_image, denoised_image)
            print(f'{noise:^10} - gaussian - {sigma}: {psnr[i, 2, k]:.2f} dB.')

            # plot denoised image
            ax[k+1].imshow(denoised_image)
            ax[k+1].set_title(f'$\sigma$ = ({sigma}, {sigma})', fontsize=5)

        ax[0].imshow(original_image)
        ax[0].set_title('Original', fontsize=5)

        fig.savefig(
            f'../../result_images/{noise}_gaussian.eps',
            pad_inches=0,
            bbox_inches='tight',
        )

    ###############################################################################################
    #                                             PLOTS                                           #
    ###############################################################################################

    # plot PSNR performance by kernel size
    plt.clf()
    plt.ylabel('PSNR (dB)')
    plt.xlabel('kernel size')
    plt.xticks([3, 5, 7])
    plt.ylim([20.5, 28.5])

    colors = ['r', 'b']
    styles = ['-o', '--o']
    for i, noise in enumerate(['saltpepper', 'gaussian']):
        for j, kernel_type in enumerate(['box', 'median']):
            plt.plot(
                [3, 5, 7], psnr[i, j, :], 
                styles[j], c=colors[i], label=f'{noise} {kernel_type}',
            )
    plt.legend()    
    
    plt.savefig(
            '../../result_images/PSNR_1.eps',
            pad_inches=0.1,
            bbox_inches='tight',
            format='eps'
        )

    # plot PSNR performance by sigma
    plt.clf()
    plt.ylabel('PSRN (dB)')
    plt.xlabel('sigma')
    plt.xticks([1, 2, 3])
    plt.ylim([20.5, 28.5])

    colors = ['r', 'b']
    styles = ['-o', '--o']
    for i, noise in enumerate(['saltpepper', 'gaussian']):
        plt.plot(
            [1, 2, 3], psnr[i, 2, :], 
            styles[j], c=colors[i], label=f'{noise} gaussian',
        )
    plt.legend()
    
    plt.savefig(
            '../../result_images/PSNR_2.eps',
            pad_inches=0.1,
            bbox_inches='tight',
            format='eps'
        )