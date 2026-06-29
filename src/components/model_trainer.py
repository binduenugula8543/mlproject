import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# -----------------------------
# Trainer Class
# -----------------------------
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # -----------------------------
            # Models (NO INVALID PARAMS)
            # -----------------------------
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
                "LinearRegression": LinearRegression(),
                "XGBRegressor": XGBRegressor(eval_metric="rmse"),
                "CatBoost": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

            logging.info("Training multiple models")

            model_scores = {}

            # -----------------------------
            # Train + Evaluate manually
            # -----------------------------
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_scores[name] = score

            # -----------------------------
            # Best model selection
            # -----------------------------
            best_model_name = max(model_scores, key=model_scores.get)
            best_model_score = model_scores[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best R2 score: {best_model_score}")

            # -----------------------------
            # Safety check
            # -----------------------------
            if best_model_score < 0.6:
                raise CustomException("No good model found (R2 < 0.6)")

            # -----------------------------
            # Save model
            # -----------------------------
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            # -----------------------------
            # Final evaluation
            final_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, final_pred)

            return r2

        except Exception as e:
            raise CustomException(e, sys)