from sklearn.neighbors import NearestNeighbors
import numpy as np


import cv2
import networkx as nx
import numpy as np
image='./phot.jpg'

def find_neighbors(image,k):
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
    return [distances, indices]

find_neighbors(image, 25)

def create_graph(k, neighbors, weights):  # # of pixels, neighbors neighbors matrix,
    G = nx.Graph()
    for i in range(len(neighbors)):
        for j in range(1, k):
            G.add_edge(neighbors[i][0], neighbors[i][j], weight = weights[i][j])
    return G

def write_output(G):
    output_template = "./output_graph.gexf"

    nx.write_gexf(G,output_template)

#print(find_neighbors(image,50)[1])
#print(find_neighbors(image,5)[1][0][1])
#G = create_graph(25, find_neighbors(image,25)[1], find_neighbors(image,25)[0])

#write_output(G)

# cv2.imshow("img", img)
# cv2.waitKey(0)