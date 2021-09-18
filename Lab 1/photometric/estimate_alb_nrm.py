import math
import numpy as np

from matplotlib import image


def estimate_alb_nrm(image_stack, V, shadow_trick=True):
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked
    # up on the 3rd dimension
    # V : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in
    # solving linear equations

    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    h, w, _ = image_stack.shape

    # Create albedo and nomral arrays
    albedo = np.zeros([h, w])  # albedo (1 channel)
    normal = np.zeros([h, w, 3])  # normal (3 channels)

    # The stack is a MxNxK tensor, where each entry contains a
    # list of K brightness values (one for each image)
    for x, row in enumerate(image_stack):
        for y, values in enumerate(row):  # brightness values
            # Stack image values into a vector i
            i = np.array(values)  # Kx1 vector

            g = None
            if shadow_trick:
                # Construct the diagonal matrix I
                I = np.diag(i)  # noqa # KxK matrix

                # Solve I*V*g(x, y) = I*i, where V is a Kx3                
                g = np.linalg.lstsq(I @ V, I @ i, rcond=None)[0]
            else:
                # Solve V*g(x, y) = i, where V is a Kx3
                g, _, _, _ = np.linalg.lstsq(V, i)

            # Calculate magnitude of vector g
            magnitude_g = np.linalg.norm(g, ord=2)

            # Save |g| in albedo, and normal at index (x, y)
            albedo[x, y] = magnitude_g
            normal[x, y] = g / (magnitude_g + 1e-10) # to avoid NaN

    return albedo, normal


if __name__ == '__main__':
    # EXAMPLE USAGE
    n = 5
    image_stack = np.zeros([10, 10, n])
    scriptV = np.zeros([n, 3])
    estimate_alb_nrm(image_stack, scriptV, shadow_trick=True)
