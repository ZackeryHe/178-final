from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, coverage_error

class helper():
    def __init__(self, model=None, X_tr=None, y_tr=None, X_te=None, y_te=None):
        self.model = model
        self.X_tr = X_tr
        self.X_te = X_te
        self.y_tr = y_tr
        self.y_te = y_te

    @staticmethod
    def print_classification(y_true, y_pred):
        print(classification_report(y_true, y_pred, target_names=["T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal",
                                                            "Shirt", "Sneaker", "Bag", "Ankle boot"]))
    
    @staticmethod
    def plot_confusion_matrix(y_pred: np.array, y_true: np.array) -> None:
        cm = confusion_matrix(y_true, y_pred)
        class_names = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal",
                                                            "Shirt", "Sneaker", "Bag", "Ankle boot"]
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=class_names)
        disp.plot(xticks_rotation=45); 
        
        plt.show()

    @staticmethod
    def scale_data(X_tr: np.array, X_te: np.array) -> tuple[np.array, np.array]:
        scaler_tr = StandardScaler()
        
        scaler_tr.fit(X_tr)

        X_tr_scaled = scaler_tr.transform(X_tr)
        X_te_scaled= scaler_tr.transform(X_te)
        return X_tr_scaled, X_te_scaled
    
    @staticmethod
    def check_coverage_error(knn, svc, lr, mlp, cnn, X_te, y_te):
        """kwg should be models"""
        error = []
        models = ["KNN", "SVC", "Logistic Regression", "Multi-layer Preceptron", "Convolutional Nuerol Network"]
        error.append(coverage_error(y_te, knn.score(X_te, y_te)))
        error.append(coverage_error(y_te, svc.score(X_te, y_te)))
        error.append(coverage_error(y_te, lr.score(X_te, y_te)))
        error.append(coverage_error(y_te, mlp.score(X_te, y_te)))
        error.append(coverage_error(y_te, cnn.score(X_te, y_te)))

        fig, axes = plt.subplots(1, figsize=(8,6))
        axes.scatter(models, error)
        axes.set_title("Coverage Error Report")
        plt.show()

    @staticmethod
    def stack_ensambles(X_tr, y_tr, seed, estimators:list):
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), seed=seed)
        clf.fit(X_tr, y_tr)
        return clf
    
    

        
            
