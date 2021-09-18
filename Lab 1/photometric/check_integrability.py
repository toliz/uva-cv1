import numpy as np


def check_integrability(normals):
    # CHECK_INTEGRABILITY check the surface gradient is acceptable
    # normals: normal image of size MxNxK tensor
    # p : df/dx
    # q : df/dy
    # SE : Squared Errors of the 2 second derivatives

    # Initalization
    p = np.zeros(normals.shape[:2])  # MxN matrix
    q = np.zeros(normals.shape[:2])  # MxN matrix
    SE = np.zeros(normals.shape[:2]) # MxN matrix

    for x, row in enumerate(normals):
        for y, normal in enumerate(row):
            p[x, y] = normal[0] / (normal[2] + 1e-10)
            q[x, y] = normal[1] / (normal[2] + 1e-10)

    # Change NaN to 0
    p[p != p] = 0
    q[q != q] = 0

    for x, row in enumerate(normals):
        for y, normal in enumerate(row):
            if x < p.shape[0] - 1 and y < p.shape[1] - 1: # avoid boundaries
                SE[x, y] = ( (p[x, y+1] - p[x, y]) - (q[x+1, y] - q[x, y]) ) ** 2

    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10, 10, 3])
    check_integrability(normals)
