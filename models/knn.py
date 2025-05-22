import os
import sys
sys.path.append(os.getcwd())

from utils.mnist_reader import load_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class KNNModel:
    def __init__(self, n_neighbors: int = 5, **kwargs):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


if __name__ == "__main__":
    X_train, y_train = load_mnist('data/fashion', kind='train')
    X_test,  y_test  = load_mnist('data/fashion', kind='t10k')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)


    knn = KNNModel(n_neighbors=5)
    knn.fit(X_train, y_train)
    print("Train accuracy:", knn.score(X_train, y_train))
    print("Test accuracy: ", knn.score(X_test,  y_test))