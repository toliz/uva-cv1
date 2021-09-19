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
    SE = np.zeros(normals.shape[:2])  # MxN matrix

    h, w = p.shape

    for x in range(w):
        for y in range(h):
            normal = normals[y][x]
            p[y][x] = normal[0]/(normal[2] + 1e-10)
            q[y][x] = normal[1]/(normal[2] + 1e-10)

    # Change NaN to 0
    p[p != p] = 0
    q[q != q] = 0

    for x in range(1, w):
        for y in range(1, h):
            SE[y-1][x-1] = ((p[y][x] - p[y-1][x]) + (q[y][x] - q[y][x-1]))**2

    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10, 10, 3])
    check_integrability(normals)
