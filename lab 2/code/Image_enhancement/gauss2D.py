import numpy as np

from gauss1D import gauss1D

def gauss2D(sigma_x, sigma_y, kernel_size):
    # Calculate 1D Gaussian filter along x-, and y-axis
	G_sigma_x = gauss1D(sigma_x, kernel_size)
	G_sigma_y = gauss1D(sigma_y, kernel_size)

	return np.dot(G_sigma_x, G_sigma_y.T)


if __name__ == "__main__":
    sigma_x = 2
    sigma_y = 2
    kernel_size = 5

    convolutional_kernel_2D = gauss2D(sigma_x, sigma_y, kernel_size)
    print(
        f"gauss1D({sigma_x}, {sigma_x}, {kernel_size}):"
        f"\n{convolutional_kernel_2D} "
        f"with shape: {convolutional_kernel_2D.shape}"
    )