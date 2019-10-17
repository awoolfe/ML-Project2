from sklearn.linear_model import LogisticRegression
import numpy as np

class stackingEnsemble:
    def __init__(self, modelList):
        self.models = modelList
        self.metaModel = LogisticRegression()

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        predictions = np.array(([]), dtype = object)
        for i in range(len(self.models)):
            predictions = np.append(predictions, model.predict(X))
        self.metaModel.fit(predictions, y)


    def predict(self, X):
        predictions = np.array(([]), dtype=object)
        for model in self.models:
            predictions = np.append(predictions, model.predict(X))
        return self.metaModel.predict(predictions)