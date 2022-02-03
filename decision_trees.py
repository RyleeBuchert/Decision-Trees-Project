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
        self.pick_attribute(concat_data)

    def pick_attribute(self, input_data):
        
        feature_results = {}
        for i in self.features:
            feature_categories = np.unique(input_data[i])
            
            subset_dictionary = {}
            class_count_dictionary = {}
            for j in feature_categories:
                subset_dictionary.update({j: input_data.loc[input_data[i] == j]})
                class_count_dictionary.update({j: {}})
                for k in self.classes:
                    class_count_dictionary[j].update({k: len(subset_dictionary[j].loc[subset_dictionary[j][self.Y_name] == k])})
                class_count_dictionary[j].update({'Total': len(subset_dictionary[j])})

            loss_score = 0
            feature_results.update({i: loss_score})

if __name__ == "__main__":

    golf_data = pd.read_csv('data\\golf_data.csv')
    X_train = golf_data.drop(golf_data.columns[4], axis=1)
    Y_train = golf_data['PlayGolf']

    DT = DecisionTree()
    DT.train(X_train, Y_train)