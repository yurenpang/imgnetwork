from Graph import *
from ANCA import *
# from MST import *
from sklearn.neighbors import NearestNeighbors
import cv2
image = 'phot.png'
import pandas as pd
import numpy as np


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
            R, G, B = img[i][j]
            feature_space.append([i, j, R, G, B])

    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute').fit(feature_space)
    distances, indices = nbrs.kneighbors(feature_space)

    return [indices, distances]

#print(find_neighbors(image, 3))

def calc_euclidean(node_1, node_2):
    length = len(node_1)
    distance = 0
    for x in range(length):
        distance += pow((node_1[x] - node_2[x]), 2)
    return math.sqrt(distance)

def find_k_neighbors(vectors, node, k_nearest_neighbor_length):
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


def find_neighbor_for_trade(feature_space, k):
    # Remove -0.0
    for i in range(len(feature_space)):
        for j in range(len(feature_space[i])):
            if feature_space[i][j] == 0 and math.copysign(1, feature_space[i][j]) == -1.0:
                feature_space[i][j] = 0.0

    # Main function body
    dic_node_and_features = {}   # key is a node, value list of neighbors
    dic_node_and_distance = {}
    feature_length = len(feature_space)

    # Find neighbors and distances for each node
    for x in range(feature_length):
        neighbors_list_and_distance = find_k_neighbors(feature_space, x, k)  # k override
        neighbors = neighbors_list_and_distance[0]
        distances = neighbors_list_and_distance[1]

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

    print(np.array(indices))
    print(distances)


    # nbrs = NearestNeighbors(n_neighbors=k).fit(feature_space)
    # distances, indices = nbrs.kneighbors(np.array(feature_space))
    #
    # print(type(indices))
    # print(distances)
    #
    # print(pd.DataFrame(feature_space))
    #
    # print(feature_space)

    return np.array(indices), np.array(distances)

def build_knn_for_trade(featureSpace,c,k):
    graph=Graph(c)
    edges,weights=find_neighbor_for_trade(featureSpace,k)

    # print('edges', edges)
    # print('117: ',len(edges))
    # print('k+1: ',len(edges[0]))


    for i in range(len(edges)):
        graph.addNode(edges[i][0])
        print(edges[i][0])
        for j in range(1,k):
            graph.addNode(edges[i][j])
            graph.addEdge(edges[i,0],edges[i,j],weights[i,j])

    graph.edges.sort(key=lambda  x:x[2])

    return graph


def build_knn_graph(image,k,c):

    knn_object=find_neighbors(image,k)

    print('got knn')
    graph = Graph(c, knn_object[5],knn_object[3],knn_object[4])
    edges = knn_object[0]
    weights = knn_object[1]
    k = knn_object[2]
    print('to for')
    for i in range(len(edges)):
        graph.addNode(edges[i][0])
        for j in range(1, k):
            graph.addNode(edges[i][j])
            graph.addEdge(edges[i, 0], edges[i, j], weights[i, j])

    print('after for')
    graph.edges.sort(key=lambda x: x[2])
    print('built graph')
    return graph

k = 20

# g = Graph([1,2,3,4,5], [(1,2,5), (2,3,8), (4,5,12), (1,5,20), (2,5,25), (3,4,30)])
#


# g=build_knn_graph(image,7,300)
# g.HFSegmentation()
# g.color()

s='edges_with_id.csv'
sn='attributes.csv'

out='./tradeNode.csv'

anca=ANCA(s, sn, 0.2, 0.1)
featureSpace=anca.anca_calc()
print(featureSpace)
country_dic = anca.realName_dic
print(len(featureSpace))

for i in range(len(featureSpace)):
    print("######################## Cluster ", i + 1)
    str = ""
    for j in (featureSpace[i]):
        country = country_dic[j]
        str += country + ", "
    print(str)

# print('# coun ',len(featureSpace))
# print('feature', len(featureSpace[0]))
# nameDic=anca.get_realName()
# graph=build_knn_for_trade(featureSpace,10,10)  # k = 5 override
#
# graph.HFSegmentation()
# graph.cluster_community(nameDic,out)





