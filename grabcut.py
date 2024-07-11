import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
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
        self.means = np.zeros((N_COMPONENTS, 3))
        self.covs = np.zeros((N_COMPONENTS, 3, 3))
        self.weights = np.zeros(N_COMPONENTS)
        kMeans = KMeans(N_COMPONENTS)
        self.clusters = kMeans.fit(pixels)
        self.cluster_labels = self.clusters.labels_
        self.clusters_index(self.clusters)

    def update_model(self, pixels, clusters):
        self.pixels = pixels
        self.cluster_labels = clusters
        self.clusters_index(clusters)
        self.update_components()
        self.calc_means_cov_matrix()

    def clusters_index(self, clusters):
        self.num_clusters = len(np.unique(clusters))
        for index, new_index in enumerate(np.unique(clusters)):
            self.cluster_labels[self.cluster_labels == new_index] = index

    def update_components(self):
        self.means = np.zeros((self.num_clusters, 3))
        self.covs = np.zeros((self.num_clusters, 3, 3))
        self.weights = np.zeros(self.num_clusters)

    def calc_means_cov_matrix(self):
        for index in range(self.num_clusters):
            self.means[index] = np.mean(self.pixels[self.cluster_labels == index], axis=0)
            self.covs[index] = np.cov(self.pixels[self.cluster_labels == index].T)
            self.weights[index] = np.sum(self.cluster_labels == index) / self.pixels.shape[0]

    def highest_likelihood_component(self, pixels):
        return np.argmax(self.likelihood(pixels), axis=0)

    def likelihood(self, pixels):
        likelihoods = []
        for cluster_index in range(self.num_clusters):
            weight = self.weights[cluster_index]
            mean = self.means[cluster_index]
            cov = self.covs[cluster_index]
            pdf_value = multivariate_normal.pdf(pixels, mean=mean, cov=cov, allow_singular=True)
            likelihoods.append(weight * pdf_value)
        return likelihoods


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)
    old_energy = None

    for i in range(n_iter):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM, i)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy, old_energy):
            break
        old_energy = energy

    mask[mask == GC_PR_BGD] = GC_BGD
    mask[mask == GC_PR_FGD] = GC_FGD
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
    cluster_fgGMM = fgGMM.highest_likelihood_component(selecting_pixels(img, mask, False))
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


def calc_n_links(img):
    img_indices = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[:2])
    rows, cols = img.shape[:2]

    edges = []
    diffs = []

    for i in range(rows):
        for j in range(cols):
            neighbors = [
                (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)
            ]
            current_pixel_index = img_indices[i, j]
            for ni, nj in neighbors:
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_pixel_index = img_indices[ni, nj]
                    edges.append([current_pixel_index, neighbor_pixel_index])
                    diff = np.linalg.norm(img[i, j] - img[ni, nj])
                    diffs.append(diff)

    edges = np.array(edges)
    diffs = np.array(diffs)
    weight = 50.0 * (np.exp(-beta * np.square(diffs)))

    graph.add_edges(edges, attributes={"weight": weight})

    edges_diag = []
    diffs_diag = []

    for i in range(rows):
        for j in range(cols):
            neighbors = [
                (i + 1, j + 1), (i - 1, j - 1), (i + 1, j - 1), (i - 1, j + 1)
            ]
            current_pixel_index = img_indices[i, j]
            for ni, nj in neighbors:
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_pixel_index = img_indices[ni, nj]
                    edges_diag.append([current_pixel_index, neighbor_pixel_index])
                    diff = np.linalg.norm(img[i, j] - img[ni, nj])
                    diffs_diag.append(diff)

    edges_diag = np.array(edges_diag)
    diffs_diag = np.array(diffs_diag)
    weight = 50.0 / np.sqrt(2.0) * (np.exp(-beta * np.square(diffs_diag)))

    graph.add_edges(edges_diag, attributes={"weight": weight})

    calc_k(img)


def calc_sum_weight_of_node(node):
    return sum([edge["weight"]for edge in graph.es.select(_source=node)])


def calc_k(img):
    max_weight = 0.0
    for node in range(img.shape[0] * img.shape[1]):
        sum_weight = calc_sum_weight_of_node(node)
        if sum_weight > max_weight:
            max_weight = sum_weight
    return max_weight


def delete_t_links(source_edge, sink_edge):
    graph.delete_edges(graph.es.select(_source=source_edge))
    graph.delete_edges(graph.es.select(_source=sink_edge))


def calc_t_links(img, mask, bgGMM, fgGMM, source_edge, sink_edge):
    soft_indices = np.where((mask == GC_PR_BGD) | (mask == GC_PR_FGD))
    bg_indices = np.where(mask == GC_BGD)
    fg_indices = np.where(mask == GC_FGD)

    soft_indices_flat = np.ravel_multi_index(soft_indices, mask.shape)
    # create edges from source to fg soft pixels
    t_fg_edges = np.zeros((soft_indices_flat.size, 2), dtype=int)
    t_fg_edges[:, 0] = source_edge
    t_fg_edges[:, 1] = soft_indices_flat
    # create edges from sink to bg soft pixels
    t_bg_edges = np.zeros((soft_indices_flat.size, 2), dtype=int)
    t_bg_edges[:, 0] = sink_edge
    t_bg_edges[:, 1] = soft_indices_flat

    soft_fg_weight = -np.log(np.sum(fgGMM.likelihood(img.reshape(-1, 3)[soft_indices_flat]), axis=0))
    soft_bg_weight = -np.log(np.sum(bgGMM.likelihood(img.reshape(-1, 3)[soft_indices_flat]), axis=0))

    fg_indices_flat = np.ravel_multi_index(fg_indices, mask.shape)
    bg_indices_flat = np.ravel_multi_index(bg_indices, mask.shape)
    # create edges from source to fg_pixels
    fg_edges_to_source = np.zeros((fg_indices_flat.size, 2), dtype=int)
    fg_edges_to_source[:, 0] = source_edge
    fg_edges_to_source[:, 1] = fg_indices_flat
    # create edges from sink to bg_pixels
    bg_edges_to_sink = np.zeros((bg_indices_flat.size, 2), dtype=int)
    bg_edges_to_sink[:, 0] = sink_edge
    bg_edges_to_sink[:, 1] = bg_indices_flat

    # set weights of fg and bg edges to be K
    fg_weight = np.full(fg_indices_flat.size, K)
    bg_weight = np.full(bg_indices_flat.size, K)

    edges = np.concatenate((t_fg_edges, t_bg_edges, fg_edges_to_source, bg_edges_to_sink), axis=0)
    weights = np.concatenate((soft_fg_weight, soft_bg_weight, fg_weight, bg_weight), axis=0)

    graph.add_edges(edges, attributes={"weight": np.array(weights)})


def calculate_mincut(img, mask, bgGMM, fgGMM ,i):  # add explanation to pdf about i
    source_edge = img.shape[0] * img.shape[1]
    sink_edge = img.shape[0] * img.shape[1] + 1
    if i == 0:  # first iteration
        global graph, beta, K
        graph = igraph.Graph(img.shape[0] * img.shape[1] + 2)
        beta = calc_beta(img)
        calc_n_links(img)
        K = calc_k(img)
    else:
        delete_t_links(source_edge, sink_edge)
    calc_t_links(img, mask, bgGMM, fgGMM, source_edge, sink_edge)
    min_cut = graph.mincut(source_edge, sink_edge, "weight")
    min_cut_sets = min_cut.partition
    min_cut_sets[0].remove(source_edge)
    min_cut_sets[1].remove(sink_edge)
    energy = min_cut.value
    return min_cut_sets, energy


def update_mask(mincut_sets, mask):
    updated_mask = np.copy(mask)
    fg_set, bg_set = mincut_sets[0], mincut_sets[1]
    for i in bg_set:
        if updated_mask[i // mask.shape[1], i % mask.shape[1]] != GC_BGD:
            updated_mask[i // mask.shape[1], i % mask.shape[1]] = GC_PR_BGD
    for i in fg_set:
        updated_mask[i // mask.shape[1], i % mask.shape[1]] = GC_PR_FGD
    return updated_mask


def check_convergence(energy, prev_energy=None):
    threshold = 0.0005  # update after running
    if prev_energy is None or prev_energy == 0:
        convergence = False
    else:
        diff = np.abs((energy - prev_energy)/energy)
        convergence = diff < threshold
    return convergence


def cal_metric(predicted_mask, gt_mask):
    return 100 * calc_accuracy(predicted_mask, gt_mask), 100 * jaccard_similarity(predicted_mask, gt_mask)


def calc_accuracy(predicted_mask, gt_mask):
    correct = np.sum(predicted_mask == gt_mask)
    return correct / gt_mask.size


def jaccard_similarity(predicted_mask, gt_mask):
    intersection = np.sum(np.logical_and(predicted_mask == GC_FGD, gt_mask == GC_FGD))
    union = np.sum(np.logical_or(predicted_mask == GC_FGD, gt_mask == GC_FGD))
    return intersection/union


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
    mask, bgGMM, fgGMM = grabcut(img, rect, 10)
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
