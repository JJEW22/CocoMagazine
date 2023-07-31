import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class DataVisualization:

    def __int__(self):
        pass

    def create_allignment_chart(self, x_axis_data, y_axis_data, x_axis_title, y_axis_title, title="TEST TITLE"):
        data_dictionary = dict()
        data_dictionary[x_axis_title] = []
        data_dictionary[y_axis_title] = []
        data_dictionary['name'] = []
        names = []
        for name in x_axis_data.keys():
            names.append(name)
            data_dictionary[x_axis_title].append(x_axis_data[name])
            data_dictionary[y_axis_title].append(y_axis_data[name])
            data_dictionary['name'].append(name)
        df = pd.DataFrame(data_dictionary, index=names)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_axis_title, y=y_axis_title)
        plt.axhline(0, color='grey')
        plt.axvline(0, color='grey')
        for i in range(df.shape[0]):
            text = df.name[i]
            plt.text(x=df[x_axis_title][i] - (len(text) * 0.0125), y=df[y_axis_title][i] + 0.03, s=text,
                     fontdict=dict(color='black', size=9),)
            # bbox=dict(facecolor='yellow', alpha = 0.5))
        plt.xlim(df[x_axis_title].min() - 0.1, df[x_axis_title].max() + 0.1)  # set x limit
        plt.ylim(df[y_axis_title].min() - 0.1, df[y_axis_title].max() + 0.1)  # set y limit
        plt.title(title) # title
        plt.xlabel(x_axis_title)  # x label
        plt.ylabel(y_axis_title)  # y label


        plt.show()

    def create_distance_matrix(self, lower_triangle, diagonal, upper_triangle):
        names = diagonal.keys()
        sorted_names = sorted(names)

        distance_matrix = np.zeros((len(names), len(names)))
        image = np.zeros((len(names), len(names)))
        central_dist_color = 0
        lower_dist_color = -1
        upper_dist_color = 1
        lower_mins_dict = dict()
        upper_mins_dict = dict()
        middle_min = 1000000
        middle_max = -100000
        for name in diagonal.keys():
            lower_min = 100000
            upper_min = 100000
            if middle_min > diagonal[name]:
                middle_min = diagonal[name]
            if middle_max < diagonal[name]:
                middle_max = diagonal[name]

            for inner_name in diagonal.keys():
                lower_triangle_value = lower_triangle[name][inner_name]
                if lower_triangle_value > 0.00001 and lower_triangle_value < lower_min:
                    lower_min = lower_triangle_value

                upper_triangle_value = upper_triangle[name][inner_name]
                if upper_triangle_value > 0.00001 and upper_triangle_value < upper_min:
                    upper_min = upper_triangle_value

            lower_mins_dict[name] = lower_min
            upper_mins_dict[name] = upper_min

        print('lower dict', lower_mins_dict)
        print('upper dict', upper_mins_dict)
        print('middle min', middle_min)
        print('middle min', middle_max)
        for row_index in range(len(sorted_names)):
            for col_index in range(row_index, len(sorted_names)):
                row_name = sorted_names[col_index]
                col_name = sorted_names[row_index]

                if row_index == col_index:
                    assert (row_name == col_name)
                    distance_matrix[row_index][col_index] = diagonal[row_name]
                    color = central_dist_color
                    if diagonal[row_name] == middle_min:
                        color = central_dist_color + 0.35
                    elif diagonal[row_name] == middle_max:
                        color = central_dist_color + 0.35
                    image[row_index][col_index] = color
                else:
                    upper_triangle_value = upper_triangle[row_name][col_name]
                    lower_triangle_value = lower_triangle[row_name][col_name]

                    distance_matrix[row_index][col_index] = upper_triangle_value
                    distance_matrix[col_index][row_index] = lower_triangle_value

                    if (upper_mins_dict[row_name] == upper_triangle_value) or (upper_mins_dict[col_name] == upper_triangle_value):
                        image[row_index][col_index] = -1 * upper_dist_color / 2
                    else:
                        image[row_index][col_index] = upper_dist_color

                    if (lower_mins_dict[row_name] == lower_triangle_value) or (lower_mins_dict[col_name] == lower_triangle_value):
                        image[col_index][row_index] = lower_dist_color / 2
                    else:
                        image[col_index][row_index] = lower_dist_color

        print('image', image)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(image)

        x0 = 10
        y0 = 10
        width = 550
        height = 400

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(sorted_names)), labels=sorted_names)
        ax.set_yticks(np.arange(len(sorted_names)), labels=sorted_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(sorted_names)):
            for j in range(len(sorted_names)):
                text = ax.text(j, i, distance_matrix[i, j],
                               ha="center", va="center", color="w")

        ax.set_title("Euclidean Distances from others and origin")
        fig.tight_layout()
        plt.show()