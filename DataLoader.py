import pandas as pd
import numpy as np
from decimal import Decimal
from scipy.spatial.distance import cosine

class DataLoader:


    def __init__(self):
        pass

    def load_data(self, path):
        dataframe = pd.read_excel(path, index_col=0)
        print('dataframe', dataframe)
        test_if_this_appears = 5
        return dataframe


    def produce_word_pairs(self, dataframe: pd.DataFrame):
        pairs = dict()
        for columnName in dataframe.columns:
            pos_word, neg_word = columnName.split('/')
            pairs[columnName] = (pos_word, neg_word)

        return pairs


    def calculate_distances(self, vectors: pd.DataFrame, power, decimal_points):
        distance_matrix = dict()
        for inner_name, inner_row in vectors.iterrows():
            inner_array = np.array(inner_row)
            for outer_name, outer_row in vectors.iterrows():
                outer_array = np.array(outer_row)
                distance = self.__minkowski_distance(inner_array, outer_array, power, decimal_points)
                row = distance_matrix.setdefault(inner_name, dict())
                row[outer_name] = distance

        return distance_matrix

    def calculate_magnitudes(self, vectors, power, decimal_points):
        magnitudes = dict()
        for name, row in vectors.iterrows():
            magnitude = self.__minkowski_distance(row, np.zeros(len(row)), power, decimal_points)
            magnitudes[name] = magnitude

        return magnitudes

    def calculate_cosine_distance(self, vectors, prefactors=0, decimal_points=2):
        distance_matrix = dict()
        for inner_name, inner_row in vectors.iterrows():
            inner_array = np.array(inner_row)
            for outer_name, outer_row in vectors.iterrows():
                outer_array = np.array(outer_row)
                distance = (10**prefactors) * round(cosine(inner_array, outer_array), decimal_points)
                row = distance_matrix.setdefault(inner_name, dict())
                row[outer_name] = distance

        return distance_matrix

    def __minkowski_distance(self, x, y, p_value, decimal_points):

        # pass the p_root function to calculate
        # all the value of vector parallelly
        return (self.__p_root(sum(pow(abs(a - b), p_value)
                           for a, b in zip(x, y)), p_value, decimal_points))

    def __p_root(self, value, root, decimal_points):
        root_value = 1 / Decimal(root)
        return round(Decimal(value) **
                     Decimal(root_value), decimal_points)




