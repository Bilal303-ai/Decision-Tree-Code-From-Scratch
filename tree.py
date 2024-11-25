import numpy as np


class DecisionTree:

    def __init__(self, max_depth, criterion='gini'):
        """
        Initialize the Decision Tree with a max depth and splitting crtierion
        :param max_depth: Maximum depth of the tree
        :param criterion: Criterion to evalate splits, "gini" or "entropy"
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None


    def gini_index(self, groups, classes):
        """
        Compute the gini index of a split
        """
        total_samples = sum([len(group) for group in groups])

        weighted_gini = 0

        for group in groups:
            size = len(group)
            if size == 0:
                continue
            summation = 0
            for class_val in classes:
                summation += (([row[-1] for row in group].count(class_val)) / size) ** 2

            gini = 1 - summation

            weighted_gini += (size / total_samples) * gini
        return weighted_gini

    def entropy(self, group):
        """Compute the entropy of a group of samples"""

        total_samples = len(group)
        if total_samples <= 1:
            entropy = 0

        else:
            labels = [row[-1] for row in group]
            classes = set(labels) #unique labels

            entropy = 0
            for class_val in classes:
                label_count = labels.count(class_val)
                ratio = label_count / total_samples
                entropy += -1 * (ratio) * np.log2(ratio)
        return entropy

    def information_gain(self, parent, children):
        '''Calculate the information gain of a split'''
        parent_size = len(parent)
        parent_entropy = self.entropy(parent)
        children_entropy = 0
        for child in children:
            child_size = len(child)
            child_entropy = self.entropy(child)
            children_entropy += (child_size / parent_size) * child_entropy

        return parent_entropy - children_entropy



    def split(self, index, value, dataset):
        """
        Split a dataset based on an attribute and an attribute value
        """
        left = []
        right = []

        for row in dataset:
            if row[index] < value:
                left.append(row)

            else:
                right.append(row)

        return left, right


    def get_best_split(self, dataset):
        """
        Get the best split for a dataset
        """

        class_values = list(set(row[-1] for row in dataset))

        best_index, best_value, best_score, best_groups = 1000, 1000, 1000, None

        for index in range(len(dataset[0]) - 1): #Exclude target column
            for row in dataset:
                groups = self.split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)

                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups


        return {'index': best_index, 'value': best_value, 'groups': best_groups}


    def make_leaf_node(self, group):
        """
        Compute the class/label of a leaf node
        """

        pred = [row[-1] for row in group]

        return(max(set(pred), key=pred.count))


    def build_tree(self, node, depth):
        """
        Build the decision tree recursively
        """
        left, right = node['groups']
        del(node['groups'])

        if not left or not right:
            node['left'] = node['right'] = self.make_leaf_node(left + right)
            return

        if depth >= self.max_depth:
            node['left'], node['right'] = self.make_leaf_node(left), self.make_leaf_node(right)
            return

        if len(left) <= 1:
            node['left'] = self.make_leaf_node(left)
        else:
            node['left'] = self.get_best_split(left)
            self.build_tree(node['left'], depth+1)

        if len(right) <= 1:
            node['right'] = self.make_leaf_node(right)
        else:
            node['right'] = self.get_best_split(right)
            self.build_tree(node['right'], depth+1)


    def fit(self, X, y):
        dataset = np.hstack((X, y.reshape(-1, 1)))
        self.tree = self.get_best_split(dataset)
        self.build_tree(self.tree, 1)

    def predict_on_single_example(self, node, row):
        """
        Predict the label given the attributes
        """

        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_on_single_example(node['left'], row)

            else:
                return node['left']

        else:
            if isinstance(node['right'], dict):
                return self.predict_on_single_example(node['right'], row)

            else:
                return node['right']

    def predict(self, X):
        """
        Predict the lables for multiple examples
        """

        return [self.predict_on_single_example(self.tree, row) for row in X]
    
tree = DecisionTree(10)

parent = [[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [2, 0, 0, 0, 1],
          [1, 1, 0, 0, 1],
          [1, 2, 1, 0, 1],
          [1, 2, 1, 1, 0],
          [2, 2, 1, 1, 1],
          [0, 1, 0, 0, 0],
          [0, 2, 1, 0, 1],
          [1, 1, 1, 0, 1],
          [0, 1, 1, 1, 1],
          [2, 1, 0, 1, 1],
          [2, 0, 1, 0, 1],
          [1, 1, 0, 1, 0]]

child1 = [[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [2, 0, 0, 0, 1],
          [1, 1, 0, 0, 1],
          [0, 1, 0, 0, 0],
          [2, 1, 0, 1, 1],
          [1, 1, 0, 1, 0]]


child2 = [[1, 2, 1, 0, 1],
          [1, 2, 1, 1, 0],
          [2, 2, 1, 1, 1],
          [0, 2, 1, 0, 1],
          [1, 1, 1, 0, 1],
          [0, 1, 1, 1, 1],
          [2, 0, 1, 0, 1]]

gain = tree.information_gain(parent, [child1, child2])
print(gain)