import os
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.exception import CustomException
from src.utils import save_object,evaluate_models,model_metrics
from src.logger import logging

@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def init_model_trainer(self,train_array, test_array):
        try:
            X_train,y_train,X_test,y_test=(train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            logging.info("Split training and test input data")
            models= {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "L1 Logistic (Lasso)": LogisticRegression(penalty='l1', solver="liblinear", max_iter=1000),
                "L2 Logistic (Ridge)": LogisticRegression(penalty='l2', solver="lbfgs", max_iter=1000),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "SVC": SVC(),
                "Decision Tree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier()
            }
            logging.info("Defined the model list")
            params = {
                "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [3, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            },

            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 20],
                "criterion": ["gini", "entropy"]
            },

            "Gradient Boosting": {
                "learning_rate": [0.1, 0.05, 0.01],
                "n_estimators": [50, 100, 200],
                "subsample": [0.6, 0.8, 1.0],
                "max_depth": [3, 5]
            },

            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100]
            },

            "L1 Logistic (Lasso)": {
                "C": [0.01, 0.1, 1, 10]
            },

            "L2 Logistic (Ridge)": {
                "C": [0.01, 0.1, 1, 10]
            },

            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            },

            "SVC": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            },

            "XGBoost": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            },

            "CatBoost": {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [50, 100, 200]
            },

            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            }          
            }
            logging.info("Defined the hyperparameter list for each model")

            model_report= evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model=model_report["best_model"]
            best_model_name=model_report["best_model_name"]

            logging.info("Best model found on both training and testing dataset")
            save_object(
                            file_path=self.model_trainer_config.trained_model_file_path,
                            obj=best_model
                        ) 
            logging.info("Saved the best model object")

            y_train_pred=best_model.predict(X_train)
            y_test_pred=best_model.predict(X_test)
            logging.info("Predicted the training and testing data")
            train_acc, train_prec, train_rec, train_f1 = model_metrics(y_train, y_train_pred)
            test_acc, test_prec, test_rec, test_f1 = model_metrics(y_test, y_test_pred)
            gap = abs(train_acc - test_acc)
            logging.info(f"Model_name: {best_model_name}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}, Gap: {gap}")
            
            return best_model, train_acc, test_acc, gap
        
        except Exception as e:
            raise CustomException(e)
