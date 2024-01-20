from zenml import pipeline
from data_ingest import ingest_data
from data_clean import clean_data
from train_model import model_train
from evalute_model import model_evaluate


@pipeline(enable_cache=True)
def create_pipeline(data_path: str):
    df = ingest_data(data_path)
    x_train,x_test,y_train,y_test=clean_data(df)
    model=model_train(df)
    r2,rmse=model_evaluate(df)
