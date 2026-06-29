import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# -----------------------------
# Config
# -----------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# -----------------------------
# Transformation Class
# -----------------------------
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # -----------------------------
    # Preprocessor creation
    # -----------------------------
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # -----------------------------
    # Transformation pipeline
    # -----------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"

            # Split input and target
            input_train_df = train_df.drop(columns=[target_column])
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(columns=[target_column])
            target_test_df = test_df[target_column]

            logging.info("Applying preprocessing on datasets")

            # Fit ONLY on train
            input_train_arr = preprocessing_obj.fit_transform(input_train_df)

            # Transform ONLY on test
            input_test_arr = preprocessing_obj.transform(input_test_df)

            # Combine features + target
            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_test_arr, np.array(target_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys) 