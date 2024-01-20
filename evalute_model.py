import  logging
import pandas as pd
from zenml import step
from evaluation import RMSE,R2,MSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluate(model: RegressorMixin,x_test:pd.DataFrame,y_test:pd.DataFrame)->Tuple[
                                                       Annotated[float,"rmse"],
                                                       Annotated[float,"r2"]]:
    try:
        prediction=model.predict(x_test)
        mse=MSE().calculate_scores(y_test,prediction)
        mlflow.log_metric("mse",mse)
        r2s=R2().calculate_scores(y_test,prediction)
        mlflow.log_metric("r2s", r2s)
        rmse=RMSE().calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse", rmse)
        return r2s,rmse,mse
    except Exception as e :
        logging.error("Error In evaluting model",e)
        raise e

