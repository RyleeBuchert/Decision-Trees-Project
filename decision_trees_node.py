import pandas as pd
import numpy as np

class Node:

    # class constructor
    def __init__(
            self,
            X = None,
            Y = None,
            is_root = None,
            is_leaf = None,
            parent = None,
            depth = None,
            max_depth = None,
            min_samples = None
            ):

        # get X and Y dataframes
        self.X = X
        self.Y = Y

        # get node information
        self.is_root = is_root if is_root else 'False'
        self.is_leaf = is_leaf if is_leaf else 'False'
        self.parent = parent if parent else 'None'
        self.best_feature = None
        self.children = []

        # stopping point information
        self.max_depth = max_depth if max_depth else 5
        self.depth = depth if depth else 0
        self.min_split_samples = min_samples if min_samples else 20

        # get all classes
        self.classes = np.unique(self.Y)
        self.num_classes = len(self.classes)
        self.Y_name = self.Y.name

        # get all features
        self.features = self.X.columns.tolist()
        self.num_features = len(self.features)

        # get percent of each class in dataset
        self.concat_data = pd.concat([self.Y, self.X], axis=1)
        self.class_count_dictionary = {i: {} for i in self.classes}
        for i in self.classes:
            self.class_count_dictionary[i].update({'Count': len(self.concat_data.loc[self.concat_data[self.Y_name] == i])})
            self.class_count_dictionary[i].update({'Percent': (self.class_count_dictionary[i]['Count'] / len(self.concat_data))})

    # method to find the best feature using cross-entropy loss
    def pick_attribute(self):

        feature_results = {}
        for i in self.features:
            # get categories for each feature and number of instances
            feature_categories = np.unique(self.X[i])
            total_count = len(self.concat_data)

            remainder = 0
            subset_dictionary = {}
            subset_count_dictionary = {}
            for j in feature_categories:
                # split data into subsets for each feature category
                subset_dictionary.update({j: self.concat_data.loc[self.concat_data[i] == j]})
                subset_count_dictionary.update({j: {}})
                
                # calculate cardinality and summed remainder
                subset_count_dictionary[j].update({'Total': len(subset_dictionary[j])})
                cardinality = ( subset_count_dictionary[j]['Total'] / total_count )
                sub_remainder = 0
                for k in self.classes:
                    subset_count_dictionary[j].update({k: len(subset_dictionary[j].loc[subset_dictionary[j][self.Y_name] == k])})
                    sub_remainder += self.cross_entropy(subset_count_dictionary[j][k] / subset_count_dictionary[j]['Total'])
                remainder += (cardinality * sub_remainder)
            
            # get information gain for each attribute
            h_prior = 0
            for k in self.classes:
                h_prior += self.cross_entropy(self.class_count_dictionary[k]['Percent'])
            information_gain = h_prior - remainder
            feature_results.update({i: information_gain})

        # return feature with the highest information gain
        self.best_feature = max(feature_results, key=feature_results.get)

    # method for calculating cross-entropy value
    def cross_entropy(self, q):
        if q == 0 or q == 1:
            return 0
        else:
            return (-1 * q) * np.log2(q)

    # method to build a decision tree with X/Y training sets
    def grow_tree(self):
        
        # check stop conditions and grow tree
        if (self.depth < self.max_depth) and (len(self.concat_data) >= self.min_split_samples):
            self.pick_attribute()
            best_feature_categories = np.unique(self.X[self.best_feature])
            new_nodes_dict = {}
            feature_subsets_dict = {}
            for i in best_feature_categories:
                feature_subsets_dict.update({i: self.concat_data.loc[self.concat_data[self.best_feature] == i]})

                # if only one class value remains, max depth is reached, or feature subset too small, create a leaf node
                if (len(np.unique(feature_subsets_dict[i][self.Y_name]))==1) or (self.depth==self.max_depth) or (len(feature_subsets_dict[i])<self.min_split_samples):
                    new_nodes_dict.update({i: Node(
                                                X = feature_subsets_dict[i].drop(columns=self.Y_name),
                                                Y = feature_subsets_dict[i][self.Y_name],
                                                is_leaf = 'True',
                                                parent=self,
                                                depth=self.depth + 1,
                                                max_depth=self.max_depth,
                                                min_samples=self.min_split_samples
                                                )})
                    self.children.append(new_nodes_dict[i])                    
                
                # else, create a new decision node and continue recursive grow
                else:
                    new_nodes_dict.update({i: Node(
                                                X = feature_subsets_dict[i].drop(columns=self.Y_name),
                                                Y = feature_subsets_dict[i][self.Y_name],
                                                parent=self,
                                                depth=self.depth + 1,
                                                max_depth=self.max_depth,
                                                min_samples=self.min_split_samples
                                                )})
                    self.children.append(new_nodes_dict[i])  
                    new_nodes_dict[i].grow_tree()


class DecisionTree:

    # class constructor
    def __init__(self):
        self.root = None
    
    # method to build tree
    def build_tree(self, X, Y, max_depth, min_samples): # add hyperparameters
        if self.root is None:
            self.root = Node(is_root='True', X=X, Y=Y, max_depth=max_depth, min_samples=min_samples)
        self.root.grow_tree()


if __name__ == "__main__":

    golf_data = pd.read_csv('data\\golf_data.csv')
    X_train = golf_data.drop(columns='PlayGolf', axis=1)
    Y_train = golf_data['PlayGolf']    

    DT = DecisionTree()
    DT.build_tree(X_train, Y_train, max_depth=5, min_samples=4)
    print()