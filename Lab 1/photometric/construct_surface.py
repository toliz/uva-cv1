import numpy as np


def construct_surface(p, q, path_type='average'):
    # CONSTRUCT_SURFACE construct the surface function
    # represented as height_map
    # p : measures value of df/dx
    # q : measures value of df/dy
    # path_type: type of path to construct height_map, either 'column',
    # 'row', or 'average'
    # height_map: the reconstructed surface

    h, w = p.shape
    height_map = np.zeros([h, w])

    if path_type == 'column':
        for y in range(1, h):
            height_map[y][0] = height_map[y-1][0] + q[y][0]

        for y in range(h):
            for x in range(1, w):
                height_map[y][x] = height_map[y][x-1] + p[y][x]

    elif path_type == 'row':
        for x in range(1, w):
            height_map[0][x] = height_map[0][x-1] + p[0][x]

        for x in range(w):
            for y in range(1, h):
                height_map[y][x] = height_map[y-1][x] + q[y][x]

    elif path_type == 'average':
        # height map by integrating by columns
        height_map_col = np.zeros([h, w])

        for y in range(1, h):
            height_map_col[y][0] = height_map_col[y-1][0] + q[y][0]

        for y in range(h):
            for x in range(1, w):
                height_map_col[y][x] = height_map_col[y][x-1] + p[y][x]

        # height map by integrating by rows
        height_map_row = np.zeros([h, w])

        for x in range(1, w):
            height_map_row[0][x] = height_map_row[0][x-1] + p[0][x]

        for x in range(w):
            for y in range(1, h):
                height_map_row[y][x] = height_map_row[y-1][x] + q[y][x]

        # average
        height_map = (height_map_col + height_map_row) / 2.0

    return height_map
