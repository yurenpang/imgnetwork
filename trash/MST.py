trashclass MST:
    def __init__(self, graph):
        """The graph should contain vertices and edges"""
        self.graph = graph
        self.vertices = graph.vertices
        self.edges = graph.edges
        self.parent = {}  # Record a node and its root
        self.rank = {} # Record a node and its rank (depth)

    def init_parent_and_rank(self):
        for vertex in self.vertices:
            self.parent[vertex] = vertex
            self.rank[vertex] = 0

    def find_root(self, vertex):
        if self.parent[vertex] == vertex:
            return vertex
        else:
            return self.find_root(self.parent[vertex])

    def union_tree_set(self, vertex1, vertex2):
        xroot = self.find_root(vertex1)
        yroot = self.find_root(vertex2)

        if self.rank[xroot] > self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] < self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[xroot] = yroot
            self.rank[yroot] += 1

    def KruskalMST(self):
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



