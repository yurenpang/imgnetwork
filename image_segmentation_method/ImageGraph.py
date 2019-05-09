# from MST import *
from sklearn.neighbors import NearestNeighbors
import csv
image = 'phot.jpg'


class CountryNetwork:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = ''
        self.vectors = []

    def find_neighbors(self, vector_names, k):
        """
        Vector names should be correct columns name in the csv,
        :param vector_names: list
        :return:
        """
        with open(self.file_name, "r") as f:
            d_reader = csv.DictReader(f)
            feature_space = []

            for row in d_reader:
                temp_vector_tuple = tuple(float(row[col]) for col in vector_names)
                feature_space.append(temp_vector_tuple)

            print((feature_space))

            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)
            distances, indices = nbrs.kneighbors(feature_space)

            return [indices, distances]


c = CountryNetwork("location_gdp.csv")
print(c.find_neighbors(["2016", "LocationX", "LocationY", "LocationZ"], 10)[0][40])
