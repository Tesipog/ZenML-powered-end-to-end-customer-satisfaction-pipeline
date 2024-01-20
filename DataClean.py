import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class Datapreprocess(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp"], axis=1)
            data["product_weight_g"].fillna(method='ffill', inplace=True)
            data["product_length_cm"].fillna(method='ffill', inplace=True)
            data["product_height_cm"].fillna(method='ffill', inplace=True)
            data["product_width_cm"].fillna(method='ffill', inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            data = data.select_dtypes(include=[np.number])

            data = data.drop(["customer_zip_code_prefix", "order_item_id"], axis=1)
            return data
        except Exception as e:
            logging.error("Error while prepocessing", e)
            raise e


class Datasplit(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> tuple[Any, Any, Any, Any]:
        try:
            x = data.drop(["review score"], axis=1)
            y = data["review score"]
            x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=0.1)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("Error while splitting", e)
            raise e
class DataCleaning:
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy
    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e :
            logging.error("Error in data handling",e)
            raise  e
if __name__=="__main__":
    data=pd.read_csv(r"C:\Users\krish\PycharmProjects\python\MLops\olist_customers_dataset.csv")
    data_cleaning=DataCleaning(data,Datapreprocess())
    data_cleaning.handle_data()