import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    target_height, target_width, target_channels = im_tgt.shape
    src_bigger = match_matrix_size(im_src, target_height, target_width, target_channels, center, 3)
    mask_bigger = match_matrix_size(im_mask, target_height, target_width, target_channels, center, 2)
    laplacian_matrix = cals_laplacian_matrix(target_height, target_width, mask_bigger)
    vector_mat = construct_matrix_B(target_height, target_width, target_channels, im_tgt,
                                    src_bigger, mask_bigger, laplacian_matrix)
    im_blend = np.zeros((target_height * target_width, target_channels), dtype=np.float32)
    solve_poisson_equation(laplacian_matrix, vector_mat, im_blend, target_channels)
    return reshape_and_normalize(im_blend, target_height, target_width, target_channels)


def match_matrix_size(img, target_height, target_width, target_channels, center, dim):
    x_value, y_value = center
    x_value -= img.shape[1] // 2
    y_value -= img.shape[0] // 2
    return padding_matrix(x_value,y_value, target_height,target_width,target_channels,img,dim)


def padding_matrix(x_value, y_value, target_height, target_width, target_channels, img, dim):
    if dim == 2:
        new_img = np.zeros((target_height, target_width), dtype=np.float32)
        new_img[y_value:y_value + img.shape[0], x_value:x_value + img.shape[1]] = img
    elif dim == 3:
        new_img = np.zeros((target_height, target_width,target_channels), dtype=np.float32)
        new_img[y_value:y_value + img.shape[0], x_value:x_value + img.shape[1], :] = img
    return new_img


def cals_laplacian_matrix(target_height, target_width,mask_bigger):
    laplacian_matrix = scipy.sparse.lil_matrix((target_height*target_width,target_height*target_width))
    for row in range(target_height):
        for col in range(target_width):
            curr_pixel_idx = row*target_width+col
            laplacian_matrix[curr_pixel_idx, curr_pixel_idx] = -4
            if mask_bigger[row, col] != 0:
                if row > 0:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx-target_width] = 1
                elif row < target_height-1:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx + target_width] = 1
                if col > 0:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx - 1] = 1
                elif col < target_width-1:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx + 1] = 1
            else:
                laplacian_matrix[curr_pixel_idx, curr_pixel_idx] = 1
    return scipy.sparse.csr_matrix(laplacian_matrix)


def construct_matrix_B(target_height, target_width, target_channels, im_tgt, img_src, img_mask, laplacian_matrix):
    flat_src = reshape(img_src)
    flat_tgt = reshape(im_tgt)
    flat_mask = img_mask.flatten()
    vector_mat = np.zeros((target_height*target_width, target_channels), dtype=np.float32)
    for color in range(target_channels):
        vector_mat[:, color] = laplacian_matrix.dot(flat_src[:, color])
        for pixel in range(flat_mask.shape[0]):
            if flat_mask[pixel] == 0:
                vector_mat[pixel, color] = flat_tgt[pixel, color]
    return vector_mat


def reshape(img):
    target_height, target_width, target_channels = img.shape
    temp_img = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=-1)
    return temp_img.reshape(target_height * target_width, target_channels).astype(np.float32)


def solve_poisson_equation(laplacian_matrix,vector_mat, solved_poisson,target_channels):
    for idx in range(target_channels):
        solved_poisson[:, idx] = spsolve(laplacian_matrix, vector_mat[:, idx])


def reshape_and_normalize(im_blend,target_height, target_width, target_channels):
    im_blend = im_blend.reshape((target_height, target_width, target_channels))
    return cv2.normalize(im_blend, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/flower.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/flower.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/grass_mountains.jpeg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
