import logging
from DataClean import DataCleaning, Datapreprocess, Datasplit
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import step


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    try:
        preprocess = Datapreprocess()
        data_clean = DataCleaning(df, preprocess)
        processed_data = data_clean.handle_data()
        divide_data = Datasplit()
        data_clean = DataCleaning(processed_data, divide_data)
        x_train, x_test, y_train, y_test = data_clean.handle_data()
        logging.info("Data cleaning is done")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Error occured during cleaning data", e)
        raise e
