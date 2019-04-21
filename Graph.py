''' This it the class to build a graph
    It has vertices as a list
    It has edges as the tuple (u node, v node, associated weight)
'''


class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.graph = {}

        self.parent = {}
        self.rank = {}
        self.size = {}

    def addEdge(self, u, v, weight):
        """Add this weighted edge to the undirected graph"""
        if (u, v, weight) not in self.edges:
            self.edges.append((u, v, weight))

    def addNode(self, node):
        """Add this node to the graph object"""
        if node not in self.vertices:
            self.vertices.append(node)
            # self.parent[node] = node
            # self.rank = 0
            # self.internal_weights = 20                    # This is the k in k/|c|

    def sortEdges(self):
        self.edges.sort(key=lambda x: x[2])

    def init_parent_and_rank(self):
        """Initialize the parent of each node to itself
           Initiatize the rank of each node to node
        """
        for vertex in self.vertices:
            self.parent[vertex] = vertex
            self.rank[vertex] = 0
            self.size[vertex] = 1

    # def init_parent_rank_weight(self, k):
    #     """Initialize the parent of each node to itself
    #        Initiatize the rank of each node to node
    #        Also initialize the internal weight of each node
    #     """
    #     for vertex in self.vertices:
    #         self.parent[vertex] = vertex
    #         self.rank[vertex] = 0
    #         self.size[vertex] = 1

    def find_root(self, vertex):
        """Find the parent of each node to check disjoint sets"""
        if self.parent[vertex] == vertex:
            return vertex
        else:
            return self.find_root(self.parent[vertex])

    def union_tree_set(self, vertex1, vertex2):
        """Merge two nodes to one set"""
        xroot = self.find_root(vertex1)
        yroot = self.find_root(vertex2)

        if self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
            self.size[xroot] += self.size[yroot]
        else:
            self.parent[xroot] = yroot
            self.size[yroot] += self.size[xroot]
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[yroot] += 1

    def KruskalMST(self):
        """Find the MST based on the Kruskal's Algorithm"""
        mst = []
        e = 0
        i = 0

        self.init_parent_and_rank()

        while e < len(self.vertices) - 1:
            v1, v2, weight = self.edges[e]
            i = i + 1
            x = self.find_root(v1)
            y = self.find_root(v2)

            if x != y:
                e = e + 1
                mst.append((v1, v2, weight))
                self.union_tree_set(x, y)

        # Return the edge with the highest weight in the component
        return mst

    #    def union_tree_set(self, vertex1, vertex2):
    #     """Merge two nodes to one set"""
    #     xroot = self.find_root(vertex1)
    #     yroot = self.find_root(vertex2)
    #
    #     if self.rank[xroot] > self.rank[yroot]:
    #         self.parent[xroot] = yroot
    #     elif self.rank[xroot] < self.rank[yroot]:
    #         self.parent[yroot] = xroot
    #     else:
    #         self.parent[xroot] = yroot
    #         self.rank[yroot] += 1

    def threshold(self, c, size):
        return c/size

    def HFSegmentation(self, k):
        self.init_parent_and_rank()
        threshold = {}
        result = {}
        for i in self.vertices:
            threshold[i] = self.threshold(k, 1)

        for edge in self.edges:
            v1, v2, weight = edge

            x = self.find_root(v1)
            y = self.find_root(v2)

            if x != y:
                # print("Pair", x, y)
                # print(threshold[x], "and", threshold[y])
                if threshold[x] > weight and threshold[y] > weight:
                    self.union_tree_set(y, x)
                    threshold[x] = weight + self.threshold(self.size[x], k)
                    result.setdefault(x, []).append(edge)

        return result

