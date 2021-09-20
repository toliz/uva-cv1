import math
import numpy as np

def gauss1D(sigma , kernel_size):
	if kernel_size % 2 == 0:
		raise ValueError(
			"kernel_size must be odd, "
			"otherwise the filter will not have a "
			"center to convolve on."
		)

    # Create Filter/Kernel
	G_sigma = lambda x: (
		np.e**-(x**2/(2*(sigma**2)))
		/
		(sigma*np.sqrt(2*np.pi))
	)

	# Create the set of x values
	range_of_x_values = np.arange(
		start=np.fix(-kernel_size/2),  # Mathematically more correct then np.floor
		stop=np.fix(kernel_size/2)+1,  # Exclusive, so increase set by 1 element
		step=1
	)
	
	# Calculate the 1D Gaussian kernel Vector
	G = np.array([[G_sigma(x)] for x in range_of_x_values])

	# Normalize before returning
	return G/(math.sqrt(np.sum(G**2)) + 1e-10)


if __name__ == "__main__":
	sigma = 2
	kernel_size = 5

	convolutional_kernel_1D = gauss1D(sigma, kernel_size)
	print(
		f"gauss1D({sigma}, {kernel_size}):"
		f"\n{convolutional_kernel_1D} with shape: {convolutional_kernel_1D.shape}"
	)