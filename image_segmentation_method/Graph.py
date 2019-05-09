''' This it the class to build a graph
    It has vertices as a list
    It has edges as the tuple (u node, v node, associated weight)
'''


class Graph:
    def __init__(self, inner_k):
        self.vertices = []
        self.edges = []
        self.inner_k = inner_k   # Initialize the inner difference of individual nodes

        self.parent = {}
        self.rank = {}
        self.size = {}
        self.threshold = {}
        self.root = set()

    def addEdge(self, u, v, weight):
        """Add this weighted edge to the undirected graph"""
        if (u, v, weight) not in self.edges:
            self.edges.append((u, v, weight))

    def addNode(self, node):
        """
        Add this node to the graph object
        Initialize parameters
        """
        if node not in self.vertices:
            self.vertices.append(node)
            self.parent[node] = node
            self.root.add(node)
            self.rank[node] = 0
            self.size[node] = 1
            self.threshold[node] = self.inner_k

    def find_root(self, vertex):
        """
        Find the parent of each node to check disjoint sets
        First copy the vertex we want to find the root for
        Find the root and passing nodes to the root as well using path compression
        """
        copy = vertex
        while copy != self.parent[copy]:
            copy = self.parent[copy]

        while vertex != copy:
            p = self.parent[vertex]
            self.parent[vertex] = copy
            vertex = p
        return copy

    def union_tree_set(self, vertex1, vertex2):
        """
        Merge two nodes to one set
        Find the bigger component using self.rank(), the root is xroot
        Set the parent of the smaller component to xroot
        Adjust the components' size
        """
        xroot = self.find_root(vertex1)  # check again
        yroot = self.find_root(vertex2)

        if self.rank[yroot] > self.rank[xroot]:
            xroot, yroot = yroot, xroot

        self.parent[yroot] = xroot
        self.root.remove(yroot)
        self.size[xroot] += self.size[yroot]
        if self.rank[xroot] == self.rank[yroot]:
            self.rank[xroot] += 1
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
        """
        Partition the graph based on https://link.springer.com/article/10.1023/B:VISI.0000022288.19776.77
        Compare inner difference of a component and the difference between components
        If the difference between components are smaller, merge two components
        :return:
        """
        for edge in self.edges:
            v1, v2, weight = edge

            x = self.find_root(v1)
            y = self.find_root(v2)

            if x != y:
                if weight < min(self.threshold[x], self.threshold[y]):
                    xroot = self.union_tree_set(y, x)
                    self.threshold[xroot] = weight + self.calc_threshold(self.inner_k, self.size[xroot])
                    print(self.threshold[xroot])
        print('segmented')

    def cluster(self):
        """
        Cluster the nodes based on HFSegmentation
        :return: output dictionary, key is the root node
        value is the set of nodes in each component
        """
        output = {}
        self.HFSegmentation()
        for v in self.vertices:
            root = self.find_root(v)
            if root not in output:
                output[root] = set()
            output[root].add(v)

        return output

# g = Graph(2)
# test = ([1,2,3,4,5], [(1,2,5), (2,3,8), (4,5,12), (1,5,20), (2,5,25), (3,4,30)])
#
# for i in test[0]:
#     g.addNode(i)
#
# for i in test[1]:
#     g.addEdge(i[0], i[1], i[2])
#
# print(g.cluster())

#g = Graph([1,2,3,4,5], [(1,2,5), (2,3,8), (4,5,12), (1,5,20), (2,5,25), (3,4,30)])
