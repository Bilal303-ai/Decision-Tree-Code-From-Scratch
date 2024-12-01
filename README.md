# About
This repo contains the python code for Decsion tree for Binary Classification. Check the file <code>tree.py</code> to check the code.
# Example usage 1:
First, clone the repository:
```
git clone https://github.com/Bilal303-ai/Decision-Tree-Code-From-Scratch
cd Decision-Tree-Code-From-Scratch
```
Import the DecisionTree class and fit the tree on your dataset
```
>>> import numpy as np
>>> from tree import DecisionTree
>>> X_train = np.array([[0, 0, 0, 0], # All features are catagorical
                  [0, 0, 0, 1],
                  [2, 0, 0, 0],
                  [1, 1, 0, 0],
                  [1, 2, 1, 0],
                  [1, 2, 1, 1],
                  [2, 2, 1, 1],
                  [0, 1, 0, 0],
                  [0, 2, 1, 0],
                  [1, 1, 1, 0]])

>>> y_train = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
>>> tree = DecisionTree(max_depth=10, criterion='gini') # The maximum depth of the tree is 10 and we are using Gini criterion for finding the best split
>>> tree.fit(X_train, y_train)
```
### Evaluate on test dataset:
```
>>> X_test = np.array([[2, 0, 0, 0],
                  [1, 1, 0, 0],
                  [1, 2, 1, 0],
                  [1, 2, 1, 1],
                  [2, 2, 1, 1]])
>>> y_test = np.array([1, 1, 1, 0, 1])
>>> accuracy = tree.evaluate(X_test, y_test)
```
### Make prediction:
```
>>> X = np.array([[1, 2, 1, 1],
                  [2, 2, 1, 1],
                  [0, 1, 0, 0],
                  [0, 2, 1, 0],
                  [1, 1, 1, 0]])
>>> predictions = tree.predict(X)
```
# Example usage 2:
```
>>> from tree import DecisionTree
>>> X_train = np.array([[1.2, 2.3, 0], # Dataset contains both numerical and categorical features
                  [3.4, 2.4, 1],
                  [7.2, 1.2 0],
                  [6.3, 4.4, 2],
                  [7.8, 9.2, 1],
                  [10.4, 4.8, 2],)

>>> y_train = np.array([0, 0, 1, 1, 1, 0])
>>> tree = DecisionTree(max_depth=100, criterion='entropy') # we are using Entropy criterion for finding the best split
>>> tree.fit(X_train, y_train)
```

