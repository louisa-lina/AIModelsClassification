import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.class_means = None
        self.class_variances = None
        self.classes = None

    def fit(self, X, y):
        """Fit the Naive Bayes model to the training data."""
        self.classes = np.unique(y)
        self.class_priors = {c: np.mean(y == c) for c in self.classes}
        self.class_means = {c: np.mean(X[y == c], axis=0) for c in self.classes}
        self.class_variances = {c: np.var(X[y == c], axis=0) for c in self.classes}

    def predict(self, X):
        """Predict class labels for the given input data."""
        predictions = []
        for x in X:
            class_probs = {}
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.class_variances[c]))
                likelihood -= 0.5 * np.sum(((x - self.class_means[c])**2) / self.class_variances[c])
                class_probs[c] = prior + likelihood
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)
