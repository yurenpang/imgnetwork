import os
import glob
import pandas as pd
import csv
import networkx as nx

class CSVProcessor:
    def __init__(self, data_folder):
        """The input data_folder should be string
           specifying the individual countries trade data"""
        # self.obtainFiles(data_folder)
        # self.clean_trade_data()
        self.country_pairs = []
        self.create_pair_and_weight()
        self.create_network()

    def obtainFiles(self, data_folder):
        owd = os.getcwd()

        os.chdir(data_folder)
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]


        # Merge the files
        # combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        # export to csv
        combined_csv.to_csv("../combined_csv.csv", index=False, encoding='utf-8-sig')

        os.chdir(owd)

    def clean_trade_data(self):
        """
        Remove the country to 'world' trade statistics
        :return: Nothing, but clean the combined_csv file
        """
        with open("combined_csv.csv", "rb") as inp, open("combined_cleaned.csv", "wb") as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if row[12] != "World":
                    writer.writerow(row)

    def create_pair_and_weight(self):
        """
        Creates the pairs of the countries with directions
        Assign the results in a tuple list
        In each tuple, the first value is the departure country,
        the second value is the destination country,
        the third is the weight (trade amount)

        :return: Nothing
        """
        with open("combined_cleaned.csv", "r") as file:
            csvreader = csv.reader(file)
            next(csvreader)
            for row in csvreader:
                if row[7] == "Export" or row[7] == "Re-Export":
                    departure = row[9]  # Report country
                    destination = row[12]  # Partner country
                else:
                    destination = row[9]
                    departure = row[12]
                weight = int(row[31])
                trade_tuple = (departure, destination, weight)
                self.country_pairs.append(trade_tuple)
        print(self.country_pairs)

    def create_network(self):
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.country_pairs)
        output_template = "./output_directed_graph.gexf"
        nx.write_gexf(g, output_template)





CSVProcessor("./data")