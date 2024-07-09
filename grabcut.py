import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans



GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel


N_COMPONENTS = 5

class GaussianMixture:
    def __init__(self,pixels):
        self.pixels = pixels
        self.num_clusters = N_COMPONENTS
        self.means = np.zeros(N_COMPONENTS,3)
        self.covariance = np.zeros(N_COMPONENTS, 3)
        self.det = np.zeros(N_COMPONENTS)
        self.weight = np.zeros(N_COMPONENTS)
        kMeans = KMeans(n_clusters=N_COMPONENTS)
        kMeans.fit(pixels)
        clusters_index(kMeans.lables_)

    def update_model(self,pixels,clusters):
        self.pixels = pixels
        self.clusters_index(clusters)
        self.update_components()
        self.calc_means_cov_matrix()

    def clusters_index(self,clusters):
        self.cluster = clusters
        self.cluster_index = clusters
        self.num_clusters = len(np.unique(clusters))
        for index,new_index in enumerate(np.unique(clusters)):
            self.cluster[self.cluster == new_index] = index


    def update_components(self):
        self.means = np.zeros(self.num_clusters,3)
        self.covariance = np.zeros(self.num_clusters, 3)
        self.det = np.zeros(self.num_clusters)
        self.weight = np.zeros(self.num_clusters)

    def calc_means_cov_matrix(self):
        for index in range(self.num_clusters):
            data = self.pixels[self.cluster_labels == index]
            self.means[index] = np.mean(data,axis = 0)
            self.covariance[index] = np.cov(data.T)
            self.det[index] = np.linalg.det(self.covariance[index])
            self.weight[index] = np.sum(self.cluster_labels == index)/self.pixels.shape[0]


   def highest_likelihood_component(self,pixels):
       likelihoods = []
       for cluster_index in range(self.n):
           weight = self.weights[cluster_index]
           mean = self.means[cluster_index]
           cov = self.covs[cluster_index]
           pdf_value = multivariate_normal.pdf(pixels, mean=mean, cov=cov, allow_singular=True)
           likelihoods.append(weight * pdf_value)
       return np.argmax(likelihoods,axis=0)



# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    bgGMM = GaussianMixture(selecting_pixels(img,mask,true))
    fgGMM = GaussianMixture(selecting_pixels(img,mask,false))
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bgGMM.calc_means_cov_matrix()
    fgGMM.calc_means_cov_matrix()
    cluster_bgGMM = bgGMM.highest_likelihood_component(selecting_pixels(img,mask,true))
    cluster_fgGMM = bgGMM.highest_likelihood_component(selecting_pixels(img, mask, true))




    return bgGMM, fgGMM

def selecting_pixels(img,mask,bg_Flag):
    if bg_Flag:
        return img[np.logical_or(mask == GC_PR_BGD,mask==GC_BGD)]
    return img[np.logical_or(mask == GC_PR_FGD,mask==GC_FGD)]




def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
