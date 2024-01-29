import pandas as pd
import numpy as np

import os
import sys
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException
from src.logger import logging

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, Y_train, X_test, Y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                # "SVM": SVC(),
                # "K Neighbours Classifier": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression(),
                # "Gaussian NB":GaussianNB(),
                # "Random Forest Classifier": RandomForestClassifier()
                }
            
            params = {
                # "SVM":{
                #     'C': [0.1, 1, 10, 100],
                #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                #     'gamma': ['scale', 'auto', 0.1, 1, 10],
                #     'degree': [2, 3, 4],
                #     'coef0': [0.0, 0.1, 0.5, 1.0]
                # }

                # "K Neighbours Classifier":{
                #     'n_neighbors': [3, 5, 7, 9],
                #     'weights': ['uniform', 'distance'],
                #     'p': [1, 2],
                #     'algorithm': ['ball_tree', 'kd_tree', 'brute']
                # },

                "Logistic Regression":{
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
                    'max_iter': [100, 200, 300, 500, 1000]
                },

                # "Gaussian NB":{
                #     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                # },

                # "Random Forest Classifier":{
                #     'n_estimators': [50, 100, 200],
                #     'max_features': ['auto', 'sqrt', 'log2'],
                #     'max_depth': [None, 10, 20, 30],
                #     'min_samples_split': [2, 5, 10],
                #     'min_samples_leaf': [1, 2, 4],
                #     'bootstrap': [True, False],
                #     'random_state': [42]
                # }
            }
            model_report:dict = evaluate_model(X_train = X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<=0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            acc = metrics.accuracy_score(Y_test, predicted)
            return acc
        
        except Exception as e:
            raise CustomException(e,sys)