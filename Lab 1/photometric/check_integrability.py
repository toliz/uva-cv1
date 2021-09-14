import numpy as np


def check_integrability(normals):
    #  CHECK_INTEGRABILITY check the surface gradient is acceptable
    #   normals: normal image of size 512x512x3 matrix
    #   p : df/dx
    #   q : df/dy
    #   SE : Squared Errors of the 2 second derivatives

    # initalization
    p = np.zeros(normals.shape[:2])  # 512x512 matrix
    q = np.zeros(normals.shape[:2])  # 512x512 matrix
    SE = np.zeros(normals.shape[:2])  # 512x512 matrix

    """
    ================
    Your code here
    ================
    Compute p and q, where
    p measures value of df/dx
    q measures value of df/dy
    """
    for x, row in enumerate(normals):
        for y, normal in enumerate(row):
            p[x][y] = normal[0]/normal[2]
            q[x][y] = normal[1]/normal[2]

    # change NaN to 0
    p[p != p] = 0
    q[q != q] = 0

    """
    ================
    Your code here
    ================
    approximate second derivate by neighbor difference
    and compute the Squared Errors SE of the 2 second derivatives SE
    """
    for x, row in enumerate(normals):
        for y, normal in enumerate(row):
            SE[x][y] = ((1/normal[2])-(1/normal[2]))**2

    return p, q, SE


if __name__ == '__main__':
    normals = np.zeros([10, 10, 3])
    check_integrability(normals)
