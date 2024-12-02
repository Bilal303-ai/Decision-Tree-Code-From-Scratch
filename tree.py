import numpy as np

class DecisionTree:
    def __init__(self, max_depth, criterion='gini'):
        """
        Initialize the Decision Tree with a max depth and splitting crtierion
        :param max_depth: Maximum depth of the tree
        :param criterion: Criterion to evalate splits, "gini" or "entropy"
        """
        if max_depth < 1:
            raise Exception("Maximum depth of the tree should be a positive integer")

        if criterion not in ['gini', 'entropy']:
            raise Exception("Allowed values for parameter 'criterion' are 'gini' and 'entropy'")
        
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None
        
    def gini_index(self, groups, classes):
        '''Compute the gini index of a split'''
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
        '''Compute the entropy of a group of samples'''

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

    def determine_attribute_type(self, attribute_values):
        ''' Determine the type (Categorical or Numerical) of an attribute. Attribute_values is an array containing the attribute values'''
        
        try:
            attribute_values = np.array(attribute_values).astype(float)
            unique_values = np.unique(attribute_values)
            ratio = len(unique_values) / len(attribute_values)
            if ratio > 0.75:
                return 'Numerical'
            else:
                return 'Categorical'
        
        except ValueError:
            return 'Categorical'

    def split_categorical(self, attribute_values, dataset):
        '''Split a dataset based on an attribute with categorical values'''
        groups = []

        
        unique_values = np.unique(attribute_values)
        for unique_value in unique_values:
            group = []
            for i in range(len(attribute_values)):
                if attribute_values[i] == unique_value:
                    group.append(dataset[i])
            groups.append((group, unique_value))
        return groups

    def split_numerical(self, attribute, value, dataset):
        '''Split a dataset based on an attribute with numerical values'''
            
        left = []
        right = []
        for row in dataset:
            if row[attribute] < value:
                left.append(row)
            else:
                right.append(row)
        if (len(left) != 0) and (len(right) != 0):
            return [left, right]
        else:
            return left + right


    def get_best_split(self, dataset):
        '''Get the best split for a dataset'''

        class_values = list(set(row[-1] for row in dataset))
        best_attribute, best_score, best_groups = 1000, 1000, None
        best_gain = 0

        for attribute in range(len(dataset[0]) - 1): #Exclude target column
            attribute_values = [row[attribute] for row in dataset]
            attribute_type = self.determine_attribute_type(attribute_values)
            if self.criterion == 'gini': 
                if attribute_type == 'Categorical':
                    groups = self.split_categorical(attribute_values, dataset)
                    gini = self.gini_index([group[0] for group in groups], class_values)

                    if gini < best_score:
                        best_attribute, attribute_type, best_value, best_score, best_groups = attribute, 'Categorical', [group[1] for group in groups], gini, [group[0] for group in groups]

                else:
                    mean_attribute_values = []
                    j = 1
                    for i in range(len(attribute_values) - 1):
                        mean_attribute_values.append((attribute_values[i] + attribute_values[j]) / 2)
                        j += 1
                    
                    for mean_value in mean_attribute_values:
                        groups = self.split_numerical(attribute, mean_value, dataset)
                        gini = self.gini_index(groups, class_values)
                        if gini < best_score:
                            best_attribute, attribute_type, best_value, best_score, best_groups = attribute, 'Numerical', mean_value, gini, groups
            
            else:
                if attribute_type == 'Categorical':
                    groups = self.split_categorical(attribute_values, dataset)
                    info_gain = self.information_gain(dataset, [group[0] for group in groups])
                    if info_gain > best_gain:
                        best_attribute, attribute_type, best_value, best_gain, best_groups,  = attribute, 'Categorical', [group[1] for group in groups], info_gain, [group[0] for group in groups],

                else:
                    mean_attribute_values = []
                    j = 1
                    for i in range(len(attribute_values) - 1):
                        mean_attribute_values.append((attribute_values[i] + attribute_values[j]) / 2)
                        j += 1

                    for mean_value in mean_attribute_values:
                        groups = self.split_numerical(attribute, mean_value, dataset)
                        info_gain = self.information_gain(dataset, groups)
                        if info_gain > best_gain:
                            best_attribute, attribute_type, best_value, best_gain, best_groups = attribute, 'Numerical', mean_value, info_gain, groups
            
        return {'attribute': best_attribute, 'attribute type':attribute_type, 'value': best_value  ,'groups': best_groups}

    def make_decision(self, group):
        '''Decide whether to convert a group into a leaf node'''
        if len(group) == 1:
            return 'Yes'
        elif len(set([row[-1] for row in group])) == 1:
            return 'Yes'
        else:
            return 'No'

    def make_leaf_node(self, group):
        '''Compute the class/label of a leaf node'''

        pred = [row[-1] for row in group]

        return(max(set(pred), key=pred.count))


    def build_tree(self, node, depth):
        '''Build the decision tree recursively'''

        if depth >= self.max_depth:
            for i, group in enumerate(node['groups']):
                node['groups'][i] = self.make_leaf_node(group)

        else:
            for i, group in enumerate(node['groups']):
                if self.make_decision(group) == 'Yes':
                    node['groups'][i] = self.make_leaf_node(group)
                else:
                    node['groups'][i] = self.get_best_split(group)
                    self.build_tree(node['groups'][i], depth+1)

    def fit(self, X, y):
        '''
        Train a decision tree on a given dataset
        X: Numpy array of shape (num_samples, num_features)
        y: Numpy array of shape (num_samples)
        '''
        dataset = np.hstack((X, y.reshape(-1, 1)))
        self.tree = self.get_best_split(dataset)
        self.build_tree(self.tree, 1)

    def predict_on_single_example(self, node, row):
        '''Predict the label given the attributes'''

        if node['attribute type'] == 'Categorical':
            for value in node['value']:
                if row[node['attribute']] == value:
                    branch = node['value'].index(value)
            
            child_node = node['groups'][branch]

            if isinstance(child_node, dict):
                return self.predict_on_single_example(child_node, row)

            else:
                return child_node

        else:
            if row[node['attribute']] < node['value']:
                child_node = node['groups'][0]
                if isinstance(child_node, dict):
                    return self.predict_on_single_example(child_node, row)
                else:
                    return child_node
            else:
                child_node = node['groups'][1]
                if isinstance(child_node, dict):
                    return self.predict_on_single_example(child_node, row)
                else:
                    return child_node


    def predict(self, X):
        '''Predict the lables for multiple examples'''

        return [self.predict_on_single_example(self.tree, row) for row in X]

    def evaluate(self, X, y):
        predictions = self.predict(X)
        score = np.sum(predictions == y)
        accuracy = score / len(X)
        return accuracy
