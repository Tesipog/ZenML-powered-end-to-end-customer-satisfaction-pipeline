import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_actual: np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_actual: np.ndarray):
        try:
            logging.info("Calculating error")
            mse = mean_squared_error(y_true, y_actual)
            logging.info("MSE: ", mse)
        except Exception as e:
            logging.error("Error In calculating MSE")
            raise e


class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_actual: np.ndarray):
        try:
            logging.info("Calculating error")
            r2s = r2_score(y_true, y_actual)
            logging.info("MSE: ", r2s)
        except Exception as e:
            logging.error("Error In calculating R2 score")
            raise e

class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_actual: np.ndarray):
        try:
            logging.info("Calculating error")
            rmse = mean_squared_error(y_true, y_actual,squared=False)
            logging.info("RMSE: ", rmse)
        except Exception as e:
            logging.error("Error In calculating RMSE")
            raise e
