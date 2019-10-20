from sklearn.linear_model import LogisticRegression
import numpy as np

class stackingEnsemble:
    def __init__(self, modelList):
        self.models = modelList
        self.metaModel = LogisticRegression(solver = "lbfgs", multi_class= 'multinomial')
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for model in self.models:
            model.fit(X, y)
        predictions = np.empty((X.shape[0], len(self.models)), dtype='object')
        for i in range(len(self.models)):
            model_predictions = self.models[i].predict(X)
            model_predictions = np.array([np.where(self.classes == x)[0] for x in model_predictions])
            predictions[:,i] = model_predictions[:,0]
        self.metaModel.fit(predictions, y)


    def predict(self, X):
        predictions = np.empty((X.shape[0], len(self.models)), dtype='object')
        for i in range(len(self.models)):
            model_predictions = self.models[i].predict(X)
            model_predictions = np.array([np.where(self.classes == x)[0] for x in model_predictions])
            predictions[:, i] = model_predictions[:, 0]
        return self.metaModel.predict(predictions)