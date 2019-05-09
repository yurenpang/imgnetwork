from Graph import *
from ANCA import *

import cv2
import csv
image = 'phot.png'
import pandas as pd
import numpy as np
from numpy import array

class nodeNetwork:
    def __init__(self, csv_file, start_column):
        self.csv_file = csv_file
        self.start_column = start_column
        self.feature_space = []
        self.realName_dic = {}

    def create_feature_space(self):
        """The csv is the string of the csv file"""
        names = pd.read_csv(self.csv_file, sep=',')
        self.realName_dic = names['Country']
        with open(self.csv_file, "r") as file:

            csvreader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            next(csvreader)
            for row in csvreader:
                feature_of_one_row = row[self.start_column:]
                self.feature_space.append(feature_of_one_row)

    def calc_euclidean(self, node_1, node_2):
        length = len(node_1)
        distance = 0

        for x in range(length):
            distance += pow((float(node_1[x]) - float(node_2[x])), 2)
        return math.sqrt(distance)

    def find_neighbors_below_threshold(self, node, threshold):
        """
        Find the neighbors of a specific node by
        Include one if the distance is below the threshold
        """
        distances = []
        neighbors = []
        for i in range(len(self.feature_space)):
            if node != i:
                dist = self.calc_euclidean(self.feature_space[node], self.feature_space[i])
                if dist <= threshold:
                    neighbors.append(i)
                    distances.append(dist)

        return [neighbors, distances]

    def find_knn_neighbors(self, node, k):
        distances = []
        neighbors = []
        neighbors_dis = []

        for i in range(len(self.feature_space)):
            if node != i:
                dist = self.calc_euclidean(self.feature_space[node], self.feature_space[i])
                distances.append((i, dist))
        distances.sort(key=lambda tup: tup[1])

        for x in range(k):
            neighbors.append(distances[x][0])
            neighbors_dis.append(distances[x][1])

        return [neighbors, neighbors_dis]

    def find_neighbors(self, threshold, k=0, method="threshold"):
        """
        The csv is the string of the csv file
        start column is the column where the feature space start
        find the neighbors based on the Euclidean distance between feature space
        input data type example: 0 Afghanistan 50 10 90 ...
        """
        self.create_feature_space()

        # Main function body
        dic_node_and_features = {}  # key is a node, value list of neighbors
        dic_node_and_distance = {}
        feature_length = len(self.feature_space)

        # Find neighbors and distances for each node
        for x in range(feature_length):
            if method == "threshold":
                neighbors_list_and_distance = self.find_neighbors_below_threshold(x, threshold)
            else:
                neighbors_list_and_distance = self.find_knn_neighbors(x, k)

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

        return np.array(indices), np.array(distances)

    def build_knn_for_trade(self, c, threshold=0, kk=0):
        graph = Graph(c)
        if threshold == 0:
            edges, weights = self.find_neighbors(0, k=kk, method="knn")
        else:
            edges, weights = self.find_neighbors(threshold)

        # print(edges)
        # print(weights)

        for i in range(len(edges)):
            graph.addNode(edges[i][0])
            for j in range(1, len(edges[i])):
                graph.addNode(edges[i][j])
                graph.addEdge(edges[i][0], edges[i][j])

        graph.edges.sort(key=lambda x: x[2])

        return graph


# attributes_file = '../database/feature_space_combined.csv'
# process = nodeNetwork(attributes_file, 1)
# g = process.build_knn_for_trade(1, 1.6)
# g.create_network(process.realName_dic)

segment_file = '../hf_segment_source.csv'
process = nodeNetwork(segment_file, 1)
g = process.build_knn_for_trade(c=45, kk=10)
g.create_network(process.realName_dic)
g.HFSegmentation()
g.cluster_community(process.realName_dic, "./hf_communities.csv")
