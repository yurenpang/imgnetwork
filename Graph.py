''' This it the class to build a graph
    It has vertices as a list
    It has edges as the tuple (u node, v node, associated weight)
'''
import random
import cv2
import networkx as nx

class Graph:
    def __init__(self,c):
        self.vertices = []
        self.c=c
        self.edges = []

        self.parent = {}
        self.root=set()
        self.rank = {}
        self.size = {}
        self.threshold= {}

    def addEdge(self, u, v, weight=1):
        """Add this weighted edge to the undirected graph"""
        if (u, v, weight) not in self.edges:
            self.edges.append((u, v, weight))

    def addNode(self, node):
        """Add this node to the graph object"""
        if node not in self.vertices:
            self.vertices.append(node)

            self.parent[node] = node
            self.root.add(node)
            self.rank[node] = 0
            self.size[node] = 1
            self.threshold[node]=self.c

    def find_root(self, vertex):
        """Find the parent of each node to check disjoint sets"""
        c = vertex
        while c != self.parent[c]:
            c = self.parent[c]
        while vertex != c:
            p = self.parent[vertex]
            self.parent[vertex] = c
            vertex = p
        return c

    def union_tree_set(self, vertex1, vertex2):
        """Merge two nodes to one set"""
        xroot = self.find_root(vertex1) #as check again
        yroot = self.find_root(vertex2)

        if self.rank[yroot]>self.rank[xroot]:
            xroot,yroot=yroot,xroot

        self.parent[yroot]=xroot
        self.root.remove(yroot)
        self.size[xroot]+=self.size[yroot]
        if self.rank[xroot]==self.rank[yroot]:
            self.rank[xroot]+=1
        return xroot


    def calc_threshold(self, c, size):
        return c/size

    def HFSegmentation(self):
        for edge in self.edges:
            v1, v2, weight = edge

            x = self.find_root(v1)
            y = self.find_root(v2)

            if x != y:
                if weight < min(self.threshold[x],self.threshold[y]):
                    xroot=self.union_tree_set(y, x)
                    self.threshold[xroot] = weight + self.calc_threshold(self.c,self.size[xroot])
        print('segmented, check root for community.')

    def create_network(self,nameDic,node_file_name):
        #        output_template = "./knn_graph.gexf"
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.edges)
        nx.set_node_attributes(g, nameDic.to_dict(), 'realName')

        nx.write_gexf(g,node_file_name)
        print('created output file to ',node_file_name)

    def color(self,image):
        img = cv2.imread(image, 1)
        w = img.shape[1]
        c={}
        for v in self.vertices:
            root=self.find_root(v)
            if root not in c:
                c[root]=(int(random.random()*255),int(random.random()*255),int(random.random()*255))
            x=v%w
            y=v//w
            theColor=c[root]
            img[y,x]=theColor

        print('color ended')
        cv2.imshow('title',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cluster_community(self, nameDic, node_file_name):
        out_node = open(node_file_name, 'w')
        out_node.write('Id, RealName, Community\n')
        self.vertices = sorted(self.vertices)

        print(self.vertices)
        for v in self.vertices:
            root = self.find_root(v)
            out_node.write(','.join([str(v), nameDic[v], str(root)]))
            out_node.write('\n')
        print('output combined to '+node_file_name)




