import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from scipy.status import multivariate_normal
import igraph

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

N_COMPONENTS = 5
graph = igraph.Graph()
beta = 0.0
K = 0.0


class GaussianMixture:
    def __init__(self, pixels):
        self.pixels = pixels
        self.num_clusters = N_COMPONENTS
        self.means = np.zeros(N_COMPONENTS, 3)
        self.covariance = np.zeros(N_COMPONENTS, 3, 3)
        self.det = np.zeros(N_COMPONENTS)
        self.weight = np.zeros(N_COMPONENTS)
        kMeans = KMeans(N_COMPONENTS)
        self.clusters = kMeans.fit(pixels)
        self.cluster_lables = self.clusters.lables_
        self.clusters_index()

    def update_model(self, pixels, clusters):
        self.pixels = pixels
        self.cluster_lables = clusters
        self.clusters_index(clusters)
        self.update_components()
        self.calc_means_cov_matrix()

    def clusters_index(self, clusters):
        self.num_cluster = len(np.unique(clusters))
        for index, new_index in enumerate(np.unique(clusters)):
            self.cluster_lables[self.cluster_lables == new_index] = index

    def update_components(self):
        self.means = np.zeros(self.num_clusters, 3)
        self.covariance = np.zeros(self.num_clusters, 3, 3)
        self.det = np.zeros(self.num_clusters)
        self.weight = np.zeros(self.num_clusters)

    def calc_means_cov_matrix(self):
        for index in range(self.num_clusters):
            self.means[index] = np.mean(self.pixels[self.cluster_labels == index], axis=0)
            self.covariance[index] = np.cov(self.pixels[self.cluster_labels == index].T)
            self.det[index] = np.linalg.det(self.covariance[index])
            self.weight[index] = np.sum(self.cluster_labels == index) / self.pixels.shape[0]

    def highest_likelihood_component(self, pixels):
        likelihoods = []
        for cluster_index in range(self.n):
            weight = self.weights[cluster_index]
            mean = self.means[cluster_index]
            cov = self.covs[cluster_index]
            pdf_value = multivariate_normal.pdf(pixels, mean=mean, cov=cov, allow_singular=True)
            likelihoods.append(weight * pdf_value)
        return np.argmax(likelihoods, axis=0)


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
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM, i)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    bgGMM = GaussianMixture(selecting_pixels(img, mask, True))
    fgGMM = GaussianMixture(selecting_pixels(img, mask, False))
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    bgGMM.calc_means_cov_matrix()
    fgGMM.calc_means_cov_matrix()
    cluster_bgGMM = bgGMM.highest_likelihood_component(selecting_pixels(img, mask, True))
    cluster_fgGMM = bgGMM.highest_likelihood_component(selecting_pixels(img, mask, False))
    bgGMM.update_model(selecting_pixels(img, mask, True), cluster_bgGMM)
    fgGMM.update_model(selecting_pixels(img, mask, False), cluster_fgGMM)
    return bgGMM, fgGMM


def selecting_pixels(img, mask, bg_Flag):
    if bg_Flag:
        return img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)]
    return img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)]


def calc_beta(img):
    squared_diffs = np.concatenate([np.square(img[:, 1:] - img[:, :-1]).sum(axis=2).flatten(),
    np.square(img[1:, :] - img[:-1, :]).sum(axis=2).flatten(),
    np.square(img[1:, :-1] - img[:-1, 1:]).sum(axis=2).flatten(),
    np.square(img[:-1, :-1] - img[1:, 1:]).sum(axis=2).flatten(),
    np.square(img[:-1, :] - img[1:, :]).sum(axis=2).flatten(),
    np.square(img[:, :-1] - img[:, 1:]).sum(axis=2).flatten(),
    np.square(img[1:, 1:] - img[:-1, :-1]).sum(axis=2).flatten(),
    np.square(img[:-1, :-1] - img[1:, 1:]).sum(axis=2).flatten(),
    ])
    return 1.0 / 2.0 * np.mean(np.pow(squared_diffs, 2))


def calc_N_links(img):
    pass  # todo


def calc_sum_weight_of_node(node):
    return sum([edge["weight"]for edge in graph.es.select(_source=node)])


def calc_K(img):
    max_weight = 0.0
    for node in range(img.shape[0] * img.shape[1]):
        sum_weight = calc_sum_weight_of_node(node)
        if sum_weight > max_weight:
            max_weight = sum_weight
    return max_weight


def delete_t_links(img):
    pass  # todo


def calc_t_links(img):
    pass  # todo


def calc_energy(min_cut, source_edge, sink_edge):
    min_cut.partition[0].remove(source_edge)
    min_cut.partition[1].remove(sink_edge)
    return min_cut.value


def calculate_mincut(img, mask, bgGMM, fgGMM ,i):  # add explanation to pdf about i
    # TODO: implement energy (cost) calculation step and mincut
    source_edge = img.shape[0] * img.shape[1]
    sink_edge = img.shape[0] * img.shape[1] + 1
    if i == 0:  # first iteration
        global graph
        graph = igraph.Graph(img.shape[0] * img.shape[1] + 2)
        calc_beta(img)
        calc_N_links(img)
        calc_K(img)
    else:
        delete_t_links(img)
    calc_t_links(img)
    min_cut = graph.mincut(source_edge, sink_edge, "weight")
    energy = calc_energy(min_cut, source_edge, sink_edge)
    return min_cut.partition, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    updated_mask = np.copy(mask)
    fg_set, bg_set = set(mincut_sets[0]), set(mincut_sets[1])
    for i in bg_set:
        if updated_mask[i // mask.shape[1], i % mask.shape[1]] != GC_BGD:
            updated_mask[i // mask.shape[1], i % mask.shape[1]] = GC_PR_BGD
    for i in fg_set:
        updated_mask[i // mask.shape[1], i % mask.shape[1]] = GC_PR_FGD
    return updated_mask


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
        rect = tuple(map(int, args.rect.split(',')))

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
