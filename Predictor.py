# importing sys
import sys
from docarray.documents import TextDoc
from OpenAPIFunction import OpenAPIFunction
import numpy as np
import time



# adding Folder_2/subfolder to the system path
sys.path.insert(0, './apiSemantic/semantle-docarray')

# importing the hello
import helper


class Predictor:

    def __int__(self):
        pass

    def generate_functions(self, word_pairs, axis, encodings, demo=True):
        function_pairs = dict()
        pos_axis_word = encodings[axis[0]]
        neg_axis_word = encodings[axis[1]]
        print('pos axis', pos_axis_word)
        print('neg axis', neg_axis_word)
        for col_name, pair in word_pairs.items():
            pos_word, neg_word = pair
            pos_word_encoding = encodings[pos_word]
            neg_word_encoding = encodings[neg_word]
            pos_pos_function = self.generate_function(pos_word_encoding, pos_axis_word, demo=demo)
            pos_neg_function = self.generate_function(pos_word_encoding, neg_axis_word, demo=demo)
            neg_pos_function = self.generate_function(neg_word_encoding, pos_axis_word, demo=demo, pos_word=pos_word_encoding)
            neg_neg_function = self.generate_function(neg_word_encoding, neg_axis_word, demo=demo, pos_word=pos_word_encoding)

            function_pairs[col_name] = ((pos_pos_function, pos_neg_function), (neg_pos_function, neg_neg_function))

        return function_pairs


    def generate_encodings(self, word_pairs, x_axis=None, y_axis=None, demo=True):
        encoded_words = dict()
        word_pairs = word_pairs.copy()
        if x_axis is not None:
            word_pairs[x_axis[0]] = x_axis[1]
        if y_axis is not None:
            word_pairs[y_axis[0]] = y_axis[1]

        for keys, word_pair in word_pairs.items():
            text_one = TextDoc(word_pair[0])
            text_two = TextDoc(word_pair[1])
            if not demo:
                helper.gpt_encode(text_one)
                time.sleep(20)
                helper.gpt_encode(text_two)
                time.sleep(20)
            encoded_words[word_pair[0]] = text_one
            encoded_words[word_pair[1]] = text_two

        return encoded_words



    def make_predictions(self, data, functions_pairings):
        predictions = dict()
        for row_name, vector in data.iterrows():
            combined_value = 0
            for key, value in vector.items():
                pos_functions, neg_functions = functions_pairings[key]
                print(key)
                total_for_section = 0
                total_for_section += pos_functions[0].calculate(value)
                total_for_section += -1 * pos_functions[1].calculate(value)
                total_for_section += neg_functions[0].calculate(-1 * value)
                total_for_section += -1 * neg_functions[1].calculate(-1 * value)

                print('total for section', total_for_section, total_for_section / value)
                combined_value += total_for_section

            combined_value = combined_value / len(vector)
            predictions[row_name] = combined_value


        return predictions

    def generate_function(self, word_from, word_to, demo=True, pos_word=None):
        return OpenAPIFunction(word_from, word_to, demo=demo, pos_word=pos_word)

    def standardize(self, predictions, scale=1):
        standardized = dict()
        values = list(predictions.values())
        print(values)
        average = np.average(values)
        deviation = np.std(values)

        for name, prediction in predictions.items():
            standardized[name] = scale * (prediction - average) / (2 * deviation)

        return standardized

    def transform_to_gausian_functions(self, function_pairings, conversion_ratio=1):
        functions = []
        function_values = []
        for key, function_pair in function_pairings.items():
            for i in [0, 1]:
                for j in [0, 1]:
                    function = function_pair[i][j]
                    functions.append(function)
                    function_values.append(function.distance)


        average = np.average(function_values)
        std_div = np.std(function_values)


        for function in functions:
            previous_contribution = function.distance * (1 - conversion_ratio)
            gaused_contribution = (function.distance - average) / (3 * std_div) * conversion_ratio
            function.distance = previous_contribution + gaused_contribution




    def magnitude_shift(self, magnitudes, average_shift=-0.1, scaling_factor=1):
        print(magnitudes)
        magnitude_values = list(magnitudes.values())
        average = np.average(magnitude_values)
        std = np.std(magnitude_values)

        shifts = dict()
        print('avg', average, 'std', std)
        for name, magnitude in magnitudes.items():
            shifts[name] = (scaling_factor * (float(magnitude) - float(average)) / (2 * float(std))) + average_shift

        return shifts




