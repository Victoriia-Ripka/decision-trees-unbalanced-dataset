from sklearn.tree import DecisionTreeClassifier
import numpy as np


class CARTModel:
    """Binary decision tree using Gini impurity (CART algorithm)."""

    def __init__(self, max_depth=None, min_samples_leaf=1, random_state=None):
        self._clf = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CARTModel":
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
