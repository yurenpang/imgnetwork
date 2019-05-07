from Graph import *
from ANCA import *
# from MST import *
from sklearn.neighbors import NearestNeighbors
import cv2
image = 'phot.png'
import pandas as pd


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

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)
    distances, indices = nbrs.kneighbors(feature_space)

    return [indices, distances, k, h, w, img]


def find_neighbor_for_trade(feature_space,k):

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)
    distances, indices = nbrs.kneighbors(feature_space)

    feature_space=pd.DataFrame(feature_space)
    print(feature_space)
    return indices,distances

def build_knn_for_trade(featureSpace,c,k):
    graph=Graph(c)
    edges,weights=find_neighbor_for_trade(featureSpace,k)

    print('edges', edges)
    print('117: ',len(edges))
    print('k+1: ',len(edges[0]))


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

anca=ANCA(s,sn,0.2,0.1)
featureSpace=anca.anca_calc()

print('# coun ',len(featureSpace))
print('feature', len(featureSpace[0]))
nameDic=anca.get_realName()
graph=build_knn_for_trade(featureSpace,2,3)

graph.HFSegmentation()
#graph.cluster_community(nameDic,out)





