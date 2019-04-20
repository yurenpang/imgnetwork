from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2
import networkx as nx
import numpy as np
import bisect
image='./phot.jpg'


from collections import defaultdict

class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.graph = {}

        # Kruskal Algorithm
        self.parent = {}
        self.rank = {}

    def addEdge(self, u, v, weight):
        # Undirected
        if (u, v, weight) not in self.edges:
            self.edges.append((u, v, weight))

    def addNode(self, node):
        if node not in self.vertices:
            self.vertices.append(node)
            self.parent[node] = node
            self.rank[node] = 0

    def sortEdges(self):
        # def getKey(item):
        #     return item[2]
        self.edges.sort(key = lambda x: x[2])


    #####################################################################
    def find(self, vertex):
        c = vertex
        while c != self.parent[c]:
            c = self.parent[c]
        while vertex != c:
            p = self.parent[vertex]
            self.parent[vertex] = c
            vertex = p
        return c
        # if parent[vertex] != vertex:
        #     parent[vertex] = self.find(parent[vertex])
        # return parent[vertex]

    def join(self, x, y):
        rootX, rootY = self.find(x), self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            else:
                self.parent[rootX] = rootY
            if self.rank[rootX] == self.rank[rootY]:
                self.rank[rootY] += 1

    def KruskalMST(self):
        '''To use Kruskal MST, the sortEdges'''
        self.sortEdges()

        print(self.edges)

        mst = set()
        for edge in self.edges:
            v1, v2, weight = edge
            if self.find(v1) != self.find(v2):
                self.join(v1, v2)
                mst.add(edge)

        return mst


'''
K-NN Algorithm 
'''
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

    return graph

knn = find_neighbors(image, 3)
g = build_knn_graph(knn)

#g.sortEdges()
print(g.KruskalMST())

# graph = Graph([], [])
#knn = find_neighbors(image, 5)
# edges = knn[0]
# weights = knn[1]
#
# for i in range(len(edges)):
#     graph.addNode(edges[i][0])
#     for j in range(1, 5):
#         graph.addNode(edges[i][j])
#         graph.addEdge(edges[i, 0], edges[i, j], weights[i, j])
#
# print(len(graph.vertices))
# print(len(graph.edges))



# def create_graph(k, neighbors, weights):  # # of pixels, neighbors neighbors matrix,
#     G = nx.MultiDiGraph()
#     for i in range(len(neighbors)):
#         for j in range(1, k):
#             G.add_edge(neighbors[i][0], neighbors[i][j], weight = weights[i][j])
#     return G

# def write_output(G):
#     output_template = "./output_undirected_graph.gexf"
#
#     nx.write_gexf(G,output_template)

'''

'''





