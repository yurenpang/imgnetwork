from Graph import *
from ANCA import *

import cv2
import csv
image = 'phot.png'
import pandas as pd
import numpy as np

class nodeNetwork:
    def __init__(self, csv_file, start_column):
        self.csv_file = csv_file
        self.start_column = start_column
        self.feature_space = []

    def create_feature_space(self):
        """The csv is the string of the csv file"""
        with open(self.csv_file, "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                feature_of_one_row = row[self.start_column:]
                self.feature_space.append(feature_of_one_row)

    def calc_euclidean(self, node_1, node_2):
        length = len(node_1)
        distance = 0
        for x in range(length):
            distance += pow((node_1[x] - node_2[x]), 2)
        return math.sqrt(distance)

    def find_neighbors_below_threshold(self, node, threshold):
        """Find the neighbors of a specific node by
        Include one if the distance is below the threshold"""
        distances = []
        neighbors = []
        for i in range(len(self.feature_space)):
            if node != i:
                dist = self.calc_euclidean(self.feature_space[node], self.feature_space[i])
                if dist <= threshold:
                    neighbors.append(i)
                    distances.append(dist)

        return [neighbors, distances]

    def find_neighbors(self, threshold):
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
            neighbors_list_and_distance = self.find_neighbors_below_threshold(x, threshold)
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

        # print(np.array(indices))
        # print(distances)
        return np.array(indices), np.array(distances)

    def build_knn_for_trade(self, c, threshold):
        graph = Graph(c)
        edges, weights = self.find_neighbors(threshold)

        for i in range(len(edges)):
            graph.addNode(edges[i][0])
            print(edges[i][0])
            for j in range(1, len(edges[i])):
                graph.addNode(edges[i][j])
                graph.addEdge(edges[i, 0], edges[i, j], weights[i, j])

        graph.edges.sort(key=lambda x: x[2])

        return graph