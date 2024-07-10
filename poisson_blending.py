import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse



def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    target_height, target_width, target_channels = im_tgt.shape
    #Ensure the source imaage and mask align properly with the target image
    src_bigger,mask_bigger = match_matrix_size(im_src,im_mask,target_height, target_width, target_channels, center)
    laplacian_matrix = cals_laplacian_matrix(target_height, target_width,mask_bigger)
    vector_mat = vector_matrix(target_height, target_width, target_channels,im_src, im_tgt, im_mask,laplacian_matrix)
    im_blend = np.zeros((target_height*target_width, target_channels), dtype = np.float32)
    solve_possion_equation(laplacian_matrix,vector_mat,im_blend,target_channels)
    return reshape_and_normalize(im_blend,target_height, target_width, target_channels)




##what we got:
##   im_blend = im_tgt
##   return im_blend



def match_matrix_size(m_src,im_mask,target_height, target_width, target_channels, center):
    x_value, y_value = center
    src = padding_matrix(x_value, y_value, target_height,target_width,target_channels,m_src,3)
    mask = padding_matrix(x_value, y_value, target_height,target_width,target_channels,im_mask,2)
    return src,mask

def padding_matrix(x_value, y_value, target_height,target_width,target_channels,img,dim):
    height,width  = img.shape[0], img.shepe[1]
    x_value-=width//2
    y_value -= height // 2
    if dim ==2:
        new_img = np.zeros((target_height,target_width),dtype = np.float32)
    elif dim==3:
        new_img = np.zeros((target_height, target_width,target_channels), dtype=np.float32)
    new_img[y_value:y_value + height, x_value:x_value + width, :] = img
    return new_img

def cals_laplacian_matrix(target_height, target_width,mask_bigger):
    laplacian_matrix = scipy.sparse.lil_matrix((target_height*target_width,target_height*target_width))
    for row in range(target_height):
        for col in range(target_width):
            curr_pixel_idx = row*target_width+col
            laplacian_matrix[curr_pixel_idx, curr_pixel_idx] = -4
            if  laplacian_matrix[row, col]!=0:
                if row> 0:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx-target_width] = 1
                elif row<target_height-1:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx + target_width] = 1
                if col>0:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx -1] = 1
                elif col<target_width-1:
                    laplacian_matrix[curr_pixel_idx, curr_pixel_idx + 1] = 1
            else:
                laplacian_matrix[curr_pixel_idx, curr_pixel_idx] = 1
    return scipy.sparse.csr_matrix(laplacian_matrix)



def vector_matrix(target_height, target_width, target_channels,im_src, im_tgt, im_mask,laplacian_matrix):
    flat_src = (im_src.reshape(-1,im_src[2])).astype(np.float32)
    flat_tgt = (im_tgt.reshape(-1,im_tgt[2])).astype(np.float32)
    flat_mask = im_mask.flatten()
    vector_mat = np.zeros((target_height*target_width,target_channels), dtype = np.float32)
    for color in range(target_channels):
        vector_mat[:, color] = laplacian_matrix.dot(flat_src[:, color])
        for idx in range(flat_mask.shape[0]):
            if flat_mask[idx]==0:
                vector_mat[idx,color] = flat_tgt[idx,color]
    return vector_mat


def solve_possion_equation(laplacian_matrix,vector_mat,solved_possion,target_channels):
    for idx in range(target_channels):
        solved_possion[:, idx] = spsolve(laplacian_matrix,vector_mat[:, idx])


def reshape_and_normalize(im_blend,target_height, target_width, target_channels):
    im_blend = im_blend.reshape((target_height, target_width, target_channels))
    return cv2.normalize(im_blend, None, 0,1.0, cv2.NORM_MINMAX, dtype =cv2.CV_32F )

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
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
