import numpy as np


def construct_surface(p, q, path_type='column'):
    # CONSTRUCT_SURFACE construct the surface function represented as height_map
    # p : measures value of df/dx
    # q : measures value of df/dy
    # path_type: type of path to construct height_map, either 'column',
    # 'row', or 'average'
    # height_map: the reconstructed surface

    h, w = p.shape
    height_map = np.zeros([h, w])

    if path_type == 'column':
        for i in range(1, h):
            height_map[i, 0] = height_map[i-1, 0] + q[i, 0]

        for i in range(h):
            for j in range(1, w):
                height_map[i, j] = height_map[i, j-1] + p[i, j]

    elif path_type == 'row':
        for j in range(1, w):
            height_map[0, j] = height_map[0, j-1] + p[0, j]

        for j in range(w):
            for i in range(1, h):
                height_map[i, j] = height_map[i-1, j] + q[i, j]
                
    elif path_type == 'average':
        # height map by integrating by columns
        height_map_col = np.zeros([h, w])

        for i in range(1, h):
            height_map_col[i, 0] = height_map_col[i-1, 0] + q[i, 0]

        for i in range(h):
            for j in range(1, w):
                height_map_col[i, j] = height_map_col[i, j-1] + p[i, j]
        
        # height map by integrating by rows
        height_map_row = np.zeros([h, w])

        for j in range(1, w):
            height_map_row[0, j] = height_map_row[0, j-1] + p[0, j]

        for j in range(w):
            for i in range(1, h):
                height_map_row[i, j] = height_map_row[i-1, j] + q[i, j]

        # average
        height_map = (height_map_col + height_map_row) / 2

    return height_map
