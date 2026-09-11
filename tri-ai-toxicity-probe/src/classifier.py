import numpy as np
import joblib
import os


class Classifier:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'trained_probe.joblib')
        self.pipeline = joblib.load(model_path)

    def predict(self, X):
        X = np.asarray(X)
        return self.pipeline.predict(X).astype(int)
