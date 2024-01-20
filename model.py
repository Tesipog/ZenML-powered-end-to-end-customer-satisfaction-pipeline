import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass


class LinearRegression(Model):
    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Erroe while Training", e)
            raise e
