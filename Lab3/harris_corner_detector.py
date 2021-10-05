import numpy as np

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from scipy.ndimage import gaussian_filter, rotate


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def harris_corner_detector(I, k=0.04, window_size=5, threshold=1e-5):
    # image preprocessing
    if I.ndim == 3:
        I = rgb2gray(I)

    I = I / I.max()

    # compute the first order derivates at every pixel
    Ix = gaussian_filter(I, 1, order=(0, 1))
    Iy = gaussian_filter(I, 1, order=(1, 0))

    # compute the products of derivatives at every pixel
    Ix2 = Ix * Ix
    Ixy = Ix * Iy
    Iy2 = Iy * Iy

    # average over local window
    Sx2 = gaussian_filter(Ix2, 2)
    Sxy = gaussian_filter(Ixy, 2)
    Sy2 = gaussian_filter(Iy2, 2)

    # find cornerness matrix
    Q = np.dstack([Sx2, Sxy, Sxy, Sy2])
    Q = Q.reshape(Sxy.shape + (2, 2))

    H = np.linalg.det(Q) - k * np.trace(Q, axis1=-2, axis2=-1) ** 2

    # find interesting points
    candidate_points = np.where(H > threshold)

    interesting_points = []
    dx = window_size // 2
    dy = window_size // 2
    for (x, y) in zip(candidate_points[0], candidate_points[1]):
        if np.all(H[x-dx:x+dx+1, y-dy:y+dy+1] <= H[x, y]):
            interesting_points += [(x, y)]

    if not interesting_points:
        return H, None, None
    else:
        interesting_points = np.array(interesting_points)
        
        return H, interesting_points[:, 1], interesting_points[:, 0]


def plot(I, r, c, filename=None):
    # image preprocessing
    if I.ndim == 3:
        I = rgb2gray(I)

    I = I / I.max()

    # compute the first order derivates at every pixel
    Ix = gaussian_filter(I, 1, order=(0, 1))
    Iy = gaussian_filter(I, 1, order=(1, 0))

    # plot
    fig, ax = plt.subplots(1, 3)
    [axis.set_axis_off() for axis in ax]

    ax[0].imshow(I, cmap='gray')
    ax[0].scatter(r, c, s=0.5, c='r')
    ax[0].set_title('Interesting points', fontsize=6)

    ax[1].imshow(Ix, cmap='gray')
    ax[1].set_title('$I_x$', fontsize=6)
    
    ax[2].imshow(Iy, cmap='gray')
    ax[2].set_title('$I_y$', fontsize=6)

    if filename:
        plt.savefig(
            filename,
            format = filename.split('.')[-1],
            bbox_inches = 'tight',
        )
    else:
        plt.tight_layout()
        plt.show()


def plot_rotation(I, angles=[45, 90], filename=None, rows=1):
    cols = int(np.ceil(len(angles) / rows))

    fig, ax = plt.subplots(rows, cols)
    ax = ax.reshape(rows, cols)
    [axis.set_axis_off() for axis in ax.flatten()]
    [axis.set_box_aspect(1) for axis in ax.flatten()]
    plt.subplots_adjust(wspace=0.1, hspace=0)


    for i, angle in enumerate(angles):
        Ir = rotate(I, angle, cval=255)
        _, r, c = harris_corner_detector(Ir)
        ax[i // cols, i % cols].imshow(Ir, cmap='gray')
        ax[i // cols, i % cols].scatter(r, c, s=0.5, c='r')
        ax[i // cols, i % cols].set_title(f'{angle}°', fontsize=6)

    if filename:
        plt.savefig(
            filename,
            format = filename.split('.')[-1],
            bbox_inches = 'tight',
        )
    else:
        plt.tight_layout()
        plt.show()


def plot_threshold(I, thresholds=[1e-6, 1e-5, 1e-4, 1e-3], filename=None, rows=1):
    cols = int(np.ceil(len(thresholds) / rows))

    fig, ax = plt.subplots(rows, cols)
    ax = ax.reshape(rows, cols)
    [axis.set_axis_off() for axis in ax.flatten()]
    #[axis.set_box_aspect(1) for axis in ax.flatten()]

    for i, threshold in enumerate(thresholds):
        _, r, c = harris_corner_detector(I, threshold=threshold)
        ax[i // cols, i % cols].imshow(I, cmap='gray')
        ax[i // cols, i % cols].scatter(r, c, s=0.5, c='r')
        ax[i // cols, i % cols].set_title(r'$R_{th} = $' + f'{threshold}', fontsize=6)

    if filename:
        plt.savefig(
            filename,
            format = filename.split('.')[-1],
            bbox_inches = 'tight',
        )
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # main part
    for img in ['toy/0001.jpg', 'doll/0200.jpg']:
        I = mpimg.imread(f'images/{img}')
        img = img.split('/')[0]

        # plot interesting points for default parameters
        _, r, c = harris_corner_detector(I)
        plot(I, r, c, filename=f'figures/1/{img}.eps')

        # try different thresholds
        plot_threshold(
            I,
            thresholds = [1e-6, 1e-5, 1e-4, 1e-3],
            filename = f'figures/1/{img}_threshold.eps',
            rows = 1,
        )

        # try different rotations
        plot_rotation(
            I,
            angles = np.arange(0, 360, 45),
            filename = f'figures/1/{img}_rotation.eps',
            rows = 2,
        )
