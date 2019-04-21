''' This it the class to build a graph
    It has vertices as a list
    It has edges as the tuple (u node, v node, associated weight)
'''


class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.graph = {}

    def addEdge(self, u, v, weight):
        # Undirected
        if (u, v, weight) not in self.edges:
            self.edges.append((u, v, weight))

    def addNode(self, node):
        if node not in self.vertices:
            self.vertices.append(node)

    def sortEdges(self):
        self.edges.sort(key=lambda x: x[2])


