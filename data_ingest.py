import logging
import pandas as pd
from zenml import  step
class  DataIngest():
    def __init__(self,data_path:str):
        self.data_path=data_path
    def get_data(self):
        logging.info("Data from ",self.data_path)
        return pd.read_csv(self.data_path)
@step
def ingest_data(data_path:str):
    try:
        data=DataIngest(data_path)
        return data.get_data()
    except Exception as e :
        logging.error("Error while ingesting data ",e)
        raise e



