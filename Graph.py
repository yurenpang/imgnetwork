''' This it the class to build a graph
    It has vertices as a list
    It has edges as the tuple (u node, v node, associated weight)
'''
import random
import cv2

class Graph:
    def __init__(self,image, h,w,c):
        self.image=image
        self.h=h
        self.w=w
        self.vertices = []
        self.c=c
        self.edges = []

        self.parent = {}
        self.rank = {}
        self.size = {}
        self.threshold= {}

    def addEdge(self, u, v, weight):
        """Add this weighted edge to the undirected graph"""
        if (u, v, weight) not in self.edges:
            self.edges.append((u, v, weight))

    def addNode(self, node):
        """Add this node to the graph object"""
        if node not in self.vertices:
            self.vertices.append(node)

            self.parent[node] = node
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
        self.size[xroot]+=self.size[yroot]
        if self.rank[xroot]==self.rank[yroot]:
            self.rank[xroot]+=1
        return xroot

    # def KruskalMST(self):
    #     """Find the MST based on the Kruskal's Algorithm"""
    #     mst = []
    #     e = 0
    #     i = 0
    #
    #     self.init_parent_and_rank()
    #
    #     while e < len(self.vertices) - 1:
    #         v1, v2, weight = self.edges[e]
    #         i = i + 1 #i for what
    #         x = self.find_root(v1)
    #         y = self.find_root(v2)
    #
    #         if x != y:
    #             e = e + 1 #for what
    #             mst.append((v1, v2, weight))
    #             self.union_tree_set(x, y)
    #
    #     # Return the edge with the highest weight in the component
    #     return mst

    def calc_threshold(self, c, size):
        return c/size

    def HFSegmentation(self):
        result = {}

        for edge in self.edges:
            v1, v2, weight = edge

            x = self.find_root(v1)
            y = self.find_root(v2)

            if x != y:
                # print("Pair", x, y)
                # print(threshold[x], "and", threshold[y])
                if weight < min(self.threshold[x],self.threshold[y]):
                    xroot=self.union_tree_set(y, x)
                    self.threshold[xroot] = weight + self.calc_threshold(self.c,self.size[xroot])
                   # result.setdefault(x, []).append(edge)
        print('segmented')
    def color(self):
        c={}
        print('start color')
        for v in self.vertices:
            root=self.find_root(v)
            if root not in c:
                c[root]=(int(random.random()*255),int(random.random()*255),int(random.random()*255))
            x=v%self.w
            y=v//self.w
            theColor=c[root]
            self.image[y,x]=theColor

        print('color ended')
        cv2.imshow('title', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


