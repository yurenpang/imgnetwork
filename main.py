from Graph import *
from MST import *
from sklearn.neighbors import NearestNeighbors
import cv2
image = './phot.jpg'


def find_neighbors(image, k):
    """
    K-NN Algorithm
    """
    img = cv2.imread(image, 1)
    feature_space = []
    h = img.shape[0]
    w = img.shape[1]

    for i in range(h):
        for j in range(w):
            R = img[i][j][0]
            G = img[i][j][1]
            B = img[i][j][2]
            feature_space.append([i, j, R, G, B])

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)
    distances, indices = nbrs.kneighbors(feature_space)

    return [indices, distances, k]


def build_knn_graph(knn_object):
    graph = Graph([], [])
    edges = knn_object[0]
    weights = knn_object[1]
    k = knn_object[2]

    for i in range(len(edges)):
        graph.addNode(edges[i][0])
        for j in range(1, k):
            graph.addNode(edges[i][j])
            graph.addEdge(edges[i, 0], edges[i, j], weights[i, j])

    return graph.sortEdges()

k = 20

g = Graph([1,2,3,4,5], [(1,2,5), (2,3,8), (4,5,12), (1,5,20), (2,5,25), (3,4,30)])

mst = MST(g)
print(mst.KruskalMST())

