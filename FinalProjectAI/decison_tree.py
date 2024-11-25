import numpy as np

class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p**2)

    def _split(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def _best_split(self, X, y):
        best_gini = 1
        best_split = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gini_split = (len(y_left) / len(y) * self._gini(y_left) +
                              len(y_right) / len(y) * self._gini(y_right))
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = (feature_index, threshold)
        return best_split

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))

        best_split = self._best_split(X, y)
        if not best_split:
            return np.argmax(np.bincount(y))

        feature_index, threshold = best_split
        X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1),
        }

    def _predict_one(self, x, tree):
        if isinstance(tree, dict):
            if x[tree["feature_index"]] <= tree["threshold"]:
                return self._predict_one(x, tree["left"])
            else:
                return self._predict_one(x, tree["right"])
        else:
            return tree
