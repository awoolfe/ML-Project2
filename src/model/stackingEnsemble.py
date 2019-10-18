from sklearn.linear_model import LogisticRegression
import numpy as np

class stackingEnsemble:
    def __init__(self, modelList):
        self.models = modelList
        self.metaModel = LogisticRegression()

    def fit(self, X, y):
        classes = np.unique(y)
        for model in self.models:
            model.fit(X, y)
        predictions = np.empty((X.shape[0], len(self.models)), dtype='object')
        for i in range(len(self.models)):
            model_predictions = self.models[i].predict(X)
            model_predictions = np.array([np.where(classes == x)[0] for x in model_predictions])
            predictions[:,i] = model_predictions[:,0]
        self.metaModel.fit(predictions, y)


    def predict(self, X):
        predictions = np.array(([]), dtype=object)
        for model in self.models:
            predictions = np.append(predictions, model.predict(X))
        return self.metaModel.predict(predictions)