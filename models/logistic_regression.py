import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix, recall_score, precision_score
import os, sys
import random as rd

from sklearn.model_selection import learning_curve
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
from utils import mnist_reader

seed = 1234

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

def scale_data(X_tr: np.array, X_te: np.array) -> tuple[np.array, np.array]:
    ### YOUR CODE STARTS HERE ###
    scaler_tr = StandardScaler()
    
    scaler_tr.fit(X_tr)

    X_tr_scaled = scaler_tr.transform(X_tr)
    X_te_scaled= scaler_tr.transform(X_te)
    ###  YOUR CODE ENDS HERE  ###
    return X_tr_scaled, X_te_scaled

class logistic_regression():
    def __init__(self, X_tr, y_tr, X_te, y_te):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_te = X_te
        self.y_te = y_te
        self.best_model = None
        self.best_validation = -np.inf

    def training_set_size(self, n_data:np.array) -> None:
        tr_accuracies = []
        te_accuracies = []
        for n in n_data:
            clf = LogisticRegression(max_iter=1000)
            clf.fit(self.X_tr[:n], self.y_tr[:n])
            tr_accuracies.append(clf.score(self.X_tr[:n], self.y_tr[:n]))
            te_accuracies.append(clf.score(self.X_te, self.y_te))
        
        """
        plot the graph
        """
        figure, axes = plt.subplots(1, figsize=(5, 4))
        axes.semilogx(n_data, tr_accuracies, color="red", label="tr_accuracy")
        axes.semilogy(n_data, te_accuracies, color="green",label="te_accuracy")
        plt.xlabel('Num. Training Data Points')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.show()
    
    def evaluate_learning_curve(self, estimator: LogisticRegression, train_size: np.array):
        
        train_size_abs, train_scores, test_scores, fit_times, score_times = learning_curve(estimator, self.X_tr, self.y_tr, train_sizes= train_size)
        for train_size, cv_train_scores, cv_test_scores, cv_fit_times, cv_score_times in zip(
            train_size_abs, train_scores, test_scores, fit_times, score_times
        ):
            print(f"{train_size} samples were used to train the model")
            print(f"The average train accuracy: {cv_train_scores.mean()}")
            print(f"The average test accuracy:{cv_test_scores.mean()}")
            print(f"The fit time for the estimator: {cv_fit_times}")
            print(f"The score time for the estimator: {cv_score_times}")
    
    @staticmethod
    def plot_confusion_matrix(y_pred: np.array, y_true: np.array) -> None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot(); 
        plt.show()

    def remove_nan(self):
        nan_remove_x = self.X_tr[~(np.isnan(self.X_tr))]
        self.X_tr = nan_remove_x

    def tain_on_random_set(self, number_datasets:int, size_of_dataset:int, **params):
        """
        the function train on random data in datasets
        
        number_datasets: indicate number of random data needed for the sets
        **param: param for logistic regression
        """
        random_idx = rd.sample(np.arange(size_of_dataset), number_datasets)
        sample_datasets = np.array(self.X_tr[idx] for idx in range(size_of_dataset) if idx in random_idx)
        sample_y = np.array(self.y_tr[idx] for idx in range(size_of_dataset) if idx in random_idx)
        model = LogisticRegression(**params)
        model.fit(sample_datasets,sample_y)
        return model
    
    @staticmethod
    def recall_presicion(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        return recall, precision
    







if __name__ == '__main__':
    X_tr_scaled, X_te_scaled = scale_data(X_train, X_test)

    logistic_regression_model = logistic_regression(X_tr_scaled, y_train, X_te_scaled, y_test)
    """
    logistic_regression_model.training_set_size([100,1000,10000,20000,40000,50000])
    """
    model1 = LogisticRegression(max_iter=1000)
    model1.fit(X_tr_scaled[:20000], y_train[:20000])
    y_pred = model1.predict(X_tr_scaled)
    logistic_regression.plot_confusion_matrix(y_pred, y_train)
    