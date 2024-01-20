import logging
import pandas as pd
from zenml import step
from model import LinearRegression
from sklearn.base import RegressorMixin
from config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_train(x_train: pd.DataFrame,
                y_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_test: pd.DataFrame) -> RegressorMixin:
    try:
        model = None

        if ModelNameConfig.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            LR = LinearRegression()
            trained_lr = LR.train(x_train, y_train)
            return trained_lr
        else:
            raise ValueError("Model", ModelNameConfig.model_name, "is not supported")
    except Exception as e:
        logging.error("Error in training model", e)
        raise e
