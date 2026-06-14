# Autorzy: Viktoriia Nowotka, Paweł Łasica
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class _Node:
    # this block the creation of the __dict__ attribute which saves memory when the tree is large
    __slots__ = ("feat", "thresh", "left", "right", "value")

    def __init__(self, value=None):
        # index of the feature that is used to split the node
        self.feat = None
        # threshold value for the split of the node
        self.thresh = None
        # pointer to the left child node 
        self.left = None
        # pointer to the right child node
        self.right = None
        # prediction value  mainly defined for leafs (most frequent class)
        self.value = value


class CARTModel:
    def __init__(self, max_depth=10, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        #  mini number of samples required in each leaf after the split
        self.min_samples_leaf = min_samples_leaf
        # rng generator
        self.rng = np.random.default_rng(random_state)
        # root node
        self.root = None
        # stores uinique class labels
        self.classes_ = None
        # no. of classes
        self.n_classes_ = 0


    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARTModel":
        # finds unique labels and maps original y to values from 0 to C-1
        self.classes_, y_int = np.unique(y, return_inverse=True)
        # saves the number classes
        self.n_classes_ = len(self.classes_)
        # build the tree via recusion  
        self.root = self._build(X, y_int, 0)
        return self

    def _build(self, X, y, depth):
        # no. of samples in the node
        n_samples = len(y)
        # count each class occurence 
        counts = np.bincount(y, minlength=self.n_classes_)
        # get the most frequent class
        node_value = self.classes_[counts.argmax()]
        # create a node assigning the most frequent class as the prediction value
        node = _Node(value=node_value)

        # stopping conditions for the recursion
        # reached the maximum depth 
        # or 
        # not enough samples to satisfy min_samples_leaf on both sides
        # or 
        # node is 100% pure (all samples belong to one class)
        if (self.max_depth is not None and depth >= self.max_depth) or (n_samples < 2 * self.min_samples_leaf) or \
           (np.max(counts) == n_samples):
            # return the node without splits - leaf
            return node

        # best split vars 
        best_gini = float('inf')
        best_feat = None
        best_thresh = None

        for feat_idx in range(X.shape[1]):
            # vector of values for the current feature
            feature_values = X[:, feat_idx]
            
            # to avoid complexity O(N^2), sort the samples by the current feature values
            # retrieve via indices
            order = np.argsort(feature_values)
            X_sorted = feature_values[order]
            y_sorted = y[order]

            # init class cnt for left split
            left_counts = np.zeros(self.n_classes_, dtype=int)
            # init class cnt for right split
            right_counts = counts.copy()

            # iterativly move samples from right to left and analyse the split between them
            for i in range(1, n_samples):
                # current class of sample being moved
                c = y_sorted[i - 1]
                left_counts[c] += 1
                right_counts[c] -= 1

                # if the feature value is the same as the previous one, we cannot split here
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                # check if the feature value is the same as the previous one, we cannot split here
                if i < self.min_samples_leaf or (n_samples - i) < self.min_samples_leaf:
                    continue

                # gini sum of squares
                sum_left_sq = np.sum(left_counts ** 2)
                sum_right_sq = np.sum(right_counts ** 2)
                
                # gini impurity for the left child := 1 - sum(proba_i^2)
                gini_left = 1.0 - sum_left_sq / (i * i) # its proba value
                # ini impurity for the left child := 1 - sum(proba_i^2)
                gini_right = 1.0 - sum_right_sq / ((n_samples - i) * (n_samples - i))
                
                # weighted gini impurity := (i * gini_left + (n_samples - i) * gini_right) / n_samples
                weighted_gini = (i * gini_left + (n_samples - i) * gini_right) / n_samples

                # check if the current split is better than the best so far
                # or if it is the same, decide via coin flip for tie breaking
                # this should give us a random tie breaking instead of deterministic one
                # because the split is random and we want to avoid bias
                if weighted_gini < best_gini - 1e-9 or \
                   (abs(weighted_gini - best_gini) <= 1e-9 and self.rng.random() < 0.5):
                    
                    # save the new best results
                    best_gini = weighted_gini
                    # save the index of the best feature
                    best_feat = feat_idx
                    # threshold value is the average of the two adjacent split points
                    best_thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0

        # if no valid split is found 
        if best_feat is None:
            # become a leaf
            return node

        # create a mask for the best split on the whole vector (without sorting)
        left_mask = X[:, best_feat] < best_thresh
        
        # save the split parameters in the node's attributes
        node.feat = best_feat
        node.thresh = best_thresh
        
        # recursively build the left branch, passing the filtered data
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        # recursively build the right branch on the remaining data
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        
        # return the node with properly attached branches
        return node

    # method to predict for a vector or matrix input X
    def predict(self, X: np.ndarray) -> np.ndarray:
        # create an empty array with the same type as the original labels
        out = np.empty(len(X), dtype=self.classes_.dtype)
        
        # iterate  all feature vectors in the matrix X
        for i, x in enumerate(X):
            # start searching from the root
            node = self.root
            # while the node has a defined split (not a leaf)
            while node.feat is not None:
                # choose the left or right side based on the threshold and feature value of the sample
                node = node.left if x[node.feat] < node.thresh else node.right
            # the value read after reaching a leaf becomes our prediction
            out[i] = node.value
            
        # return the  output vector
        return out
        
    # recursive method to count the maximum depth from the selected node
    def _tree_depth(self, node):
        # if we are in a leaf, we have reached depth 1 on this path
        if node.feat is None:
            return 1
        # return 1 plus the greater depth between the left and right child
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))

    # recursive method to count the final leaves in the structure
    def _count_leaves(self, node):
        # if the node is a leaf, count it as 1
        if node.feat is None:
            return 1
        # otherwise it is the sum of leaves in the left and right subtree
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    # decorator property allows to get the depth as an attribute (without parentheses obj.depth)
    @property
    def depth(self) -> int:
        # pass the root to the method counting the depth
        return self._tree_depth(self.root)

    # decorator property allows to get the number of leaves as an attribute (without parentheses obj.n_leaves)
    @property
    def n_leaves(self) -> int:
        # pass the root to the method counting the leaves
        return self._count_leaves(self.root)

class SklearnCARTModel:
    """Binary decision tree using sklearn's DecisionTreeClassifier (Gini)."""

    def __init__(self, max_depth=10, min_samples_leaf=1, random_state=None):
        self._clf = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnCARTModel":
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    @property
    def depth(self) -> int:
        return self._clf.get_depth()

    @property
    def n_leaves(self) -> int:
        return self._clf.get_n_leaves()


MODELS = {"cart": CARTModel, "sklearn": SklearnCARTModel}
