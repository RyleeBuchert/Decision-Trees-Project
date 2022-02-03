import pandas as pd
import numpy as np

class DecisionTree:

    def __init__(self):
        self.features = []

    def train(self, X, Y):

        self.classes = np.unique(Y)
        self.num_classes = len(self.classes)
        self.Y_name = Y.name

        self.features = X.columns.tolist()
        self.num_features = len(self.features)

        concat_data = pd.concat([Y, X], axis=1)
        self.class_count_dictionary = {i: {} for i in self.classes}
        for i in self.classes:
            self.class_count_dictionary[i].update({'Count': len(concat_data.loc[concat_data[self.Y_name] == i])})
            self.class_count_dictionary[i].update({'Percent': (self.class_count_dictionary[i]['Count'] / len(concat_data))})

        best_feature = self.pick_attribute(concat_data)

    def pick_attribute(self, input_data):

        feature_results = {}
        for i in self.features:
            feature_categories = np.unique(input_data[i])
            total_count = len(input_data)

            remainder = 0
            subset_dictionary = {}
            subset_count_dictionary = {}
            for j in feature_categories:
                subset_dictionary.update({j: input_data.loc[input_data[i] == j]})
                subset_count_dictionary.update({j: {}})
                
                subset_count_dictionary[j].update({'Total': len(subset_dictionary[j])})
                cardinality = ( subset_count_dictionary[j]['Total'] / total_count )
                sub_remainder = 0
                for k in self.classes:
                    subset_count_dictionary[j].update({k: len(subset_dictionary[j].loc[subset_dictionary[j][self.Y_name] == k])})
                    sub_remainder += self.cross_entropy(subset_count_dictionary[j][k] / subset_count_dictionary[j]['Total'])
                remainder += (cardinality * sub_remainder)
            
            h_prior = 0
            for k in self.classes:
                h_prior += self.cross_entropy(self.class_count_dictionary[k]['Percent'])
            information_gain = h_prior - remainder
            feature_results.update({i: information_gain})

        return max(feature_results, key=feature_results.get)

    def cross_entropy(self, q):
        if q == 0 or q == 1:
            return 0
        else:
            return (-1 * q) * np.log2(q)


if __name__ == "__main__":

    golf_data = pd.read_csv('data\\golf_data.csv')
    X_train = golf_data.drop(golf_data.columns[4], axis=1)
    Y_train = golf_data['PlayGolf']

    DT = DecisionTree()
    DT.train(X_train, Y_train)