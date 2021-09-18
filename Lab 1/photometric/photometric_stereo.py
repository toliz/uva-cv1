import os
import cv2
import numpy as np

from estimate_alb_nrm import estimate_alb_nrm
from construct_surface import construct_surface
from check_integrability import check_integrability
from utils import load_face_images, load_syn_images, show_results

IMAGE_PATH: str = './photometrics_images/'

print('==========================')
print('Part 1: Photometric Stereo')
print('==========================\n')


def photometric_stereo(
    image_dir: str = IMAGE_PATH + 'SphereGray5/'
):
    # obtain many images in a fixed view under different illumination
    print(f'Loading images {image_dir}...\n')
    [image_stack, scriptV] = load_syn_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)

    # compute the surface gradient from the stack of imgs and light source mat
    print('Computing surface albedo and normal map...\n')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV)

    # Integrability check: is (dp/dy - dq/dx)^2 small everywhere?
    print('Integrability checking\n')
    [p, q, SE] = check_integrability(normals)

    threshold = 0.005
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan')  # for good visualization

    # compute the surface height
    height_map = construct_surface(p, q)

    # show results
    show_results(albedo, normals, height_map, SE)


def photometric_stereo_face(
    image_dir: str = IMAGE_PATH + 'yaleB02'
):
    print(f'Loading images {image_dir}...\n')
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV)

    # integrability check: is (dp/dy - dq/dx)^2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan')  # for good visualization

    # compute the surface height
    height_map = construct_surface(p, q, type='average')

    # show results
    show_results(albedo, normals, height_map, SE)


if __name__ == '__main__':
    photometric_stereo()
    # photometric_stereo_face()
