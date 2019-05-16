from Graph import *
from ANCA import *
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np


def find_neighbors_for_image(image, k):
    img = cv2.imread(image, 1)
    feature_space = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            R, G, B = img[i][j]
            feature_space.append([i, j, R, G, B])

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)
    distances, indices = nbrs.kneighbors(feature_space)

    return [indices, distances]

def calc_euclidean(node_1, node_2):
    length = len(node_1)
    distance = 0
    for x in range(length):
        distance += pow((node_1[x] - node_2[x]), 2)
    return math.sqrt(distance)


def __find_k_neighbors(vectors, node, k_nearest_neighbor_length):
    distances = []
    neighbors = []
    neighbors_dis = []
    for i in range(len(vectors)):
        if node != i:
            dist = calc_euclidean(vectors[node], vectors[i])
            distances.append((i, dist))
    distances.sort(key=lambda tup: tup[1])

    for x in range(k_nearest_neighbor_length):
        neighbors.append(distances[x][0])
        neighbors_dis.append(distances[x][1])

    return [neighbors, neighbors_dis]


def _find_neighbor_for_trade(feature_space, k):
    # Main function body
    dic_node_and_features = {}   # key is a node, value list of neighbors
    dic_node_and_distance = {}
    feature_length = len(feature_space)

    # Find neighbors and distances for each node
    for x in range(feature_length):
        neighbors_list_and_distance = __find_k_neighbors(feature_space, x, k)  # k override
        neighbors,distances = neighbors_list_and_distance

        dic_node_and_features[x] = neighbors
        dic_node_and_distance[x] = distances

    # Update the return values
    indices = []
    distances = []
    for i in range(feature_length):
        row_neighbors = dic_node_and_features.get(i)
        row_neighbors.insert(0, i)
        row_distances = dic_node_and_distance.get(i)
        row_distances.insert(0, 0)

        indices.append(row_neighbors)
        distances.append(row_distances)

    return np.array(indices), np.array(distances)

def build_graph_from_knn(featureSpace,c,k):
    edges,weights=_find_neighbor_for_trade(featureSpace,k)
    return __build_graph(edges,weights,c)

def __build_graph(edges,weights,c):
    graph = Graph(c)
    print('to for')
    for i in range(len(edges)):
        print(i)
        graph.addNode(edges[i][0])
        for j in range(1, len(edges[i])):
            graph.addNode(edges[i][j])
            graph.addEdge(edges[i,0],edges[i,j],weights[i,j])
    print('after for')

    graph.edges.sort(key=lambda  x:x[2])

    return graph


def build_knn_graph(image,k,c):
    edges, weights=find_neighbors_for_image(image,k)
    return __build_graph(edges,weights,c)

###################### Main Graph Segmentation
image = 'phot.jpg'
g=build_knn_graph(image,7,300)
g.HFSegmentation()
g.color(image)
##############################################


###################### Main K-nearest neighbor cluster
# attributeDate='hf_segment_source.csv'
# outFile='knn_node_cluster.csv'
# featureSpace=pd.read_csv(attributeDate, sep=',')
# neighbors,distances=find_neighbor_for_trade(featureSpace,7) #k for knn is 7
#
# out_node = open(outFile, 'w')
# out_node.write('Id, RealName, Community\n')
# for i in range(len(neighbors)):
#     for j in range(1,len(neighbors[0])):
#         out_node.write(','.join([str(i), str(neighbors[i][j]), str(distances[i][j])]))
#         out_node.write('\n')
# print('output pure knn cluster to '+outFile)



###################### Main - ANCA Kmean cluster
# s='acna_knn_edges.csv'
# sn='acna_knn_nodes_space.csv'
#
# out_anca_kmean='correct_anca.csv'
#
# a=ANCA(s,sn,0.3,0.2)
# cluster=a.anca_calc_kMean_cluster_to_file(out_anca_kmean) # param K is optional


##################### Main - combined cluster
# s='acna_knn_edges.csv'
# sn='acna_knn_nodes_space.csv'
# c=15
# k=20
# out_combined='cluster_combined.csv'
# ca=ANCA(s,sn,0.3,0.2)
# featureSpace=ca.anca_calc()
# g=build_graph_from_knn(featureSpace,c,k)
# g.HFSegmentation()
# g.cluster_community(ca.realName_dic,out_combined)


