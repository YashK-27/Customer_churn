import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
            categorical_columns = ['gender','SeniorCitizen','Partner','Dependents','PhoneService', 'MultipleLines', 'InternetService',
                                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                     'Contract', 'PaperlessBilling','PaymentMethod']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete")

            train_df.drop(columns=['customerID'], inplace=True)
            test_df.drop(columns=['customerID'], inplace=True)
            logging.info("Droping of customer ID column completed")

            train_df['SeniorCitizen'] = ["No" if value == 0 else "Yes" for value in train_df['SeniorCitizen']]
            test_df['SeniorCitizen'] = ["No" if value == 0 else "Yes" for value in test_df['SeniorCitizen']]
            logging.info("Transforming of seniorcitizen column completed")

            train_df['TotalCharges'] = train_df['TotalCharges'].replace('', np.nan)
            train_df['TotalCharges'] = pd.to_numeric(train_df['TotalCharges'], errors='coerce')
            test_df['TotalCharges'] = test_df['TotalCharges'].replace('', np.nan)
            test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
            logging.info("Replacing null with nan of total charges completed")

            train_df['tenure'] = [float(value) for value in train_df['tenure']]
            test_df['tenure'] = [float(value) for value in test_df['tenure']]
            logging.info("parsing datatype of tenure completed")

            train_df.replace('No internet service', 'No', inplace=True)
            test_df.replace('No internet service', 'No', inplace=True)
            logging.info("Replacing NIS with NO completed")

            train_df.replace('No phone service', 'No', inplace=True)
            test_df.replace('No phone service', 'No', inplace=True)
            logging.info('Replacing NPS with NO completed')

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Churn"
            # numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)