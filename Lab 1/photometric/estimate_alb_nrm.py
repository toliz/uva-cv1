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

    # create arrays for
    albedo = np.zeros([h, w])  # albedo (1 channel)
    normal = np.zeros([h, w, 3])  # normal (3 channels)
    """
    ================
    Your code here
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """

    # 1. The stack is a 512*512, where each (x,y) contains a
    # list of 5 values (brightness). One for each
    # image (since we have 5 images)
    for x, row in enumerate(image_stack):
        for y, values in enumerate(row):  # 2. Stack image values
            # 2. Stack image values into a vector i
            i = np.array(values)  # 5x1 vector

            # 3. Construct the diagonal matrix scriptI
            I = np.diag(i)  # noqa # 5x5 matrix

            # 4. Solve I*V*g(x, y) = I*i, where V is a 5x3
            # g = None
            # if shadow_trick:
            g = np.linalg.lstsq(np.dot(I, V), np.dot(I, i))[0]
            # else:
            #     g = np.linalg.lstsq(np.dot(I, V), np.dot(I, i))[0]

            # Calculate magnitude of vector g
            magnitude_g = math.sqrt(np.sum(g**2))

            # 5. Save |g| in albedo, and normal at index (x, y)
            albedo[x][y] = magnitude_g
            normal[x][y] = g/magnitude_g

    return albedo, normal


if __name__ == '__main__':
    # EXAMPLE USAGE
    n = 5
    image_stack = np.zeros([10, 10, n])
    scriptV = np.zeros([n, 3])
    estimate_alb_nrm(image_stack, scriptV, shadow_trick=True)
