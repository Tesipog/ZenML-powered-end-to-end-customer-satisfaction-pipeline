import numpy as np
# import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from data_clean import clean_data
from data_ingest import ingest_data
from evalute_model import model_evaluate
from train_model import model_train

docker_settings = DockerSettings(required_integrations=[MLFLOW])

import json

# from .utils import get_data_for_test
import os

import numpy as np
import pandas as pd

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

# from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])
import pandas as pd

from utils import get_data_for_test

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0.9


@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig,
) -> bool:
    return accuracy > config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
        pipeline_name: str,
        pipeline_step_name: str,
        running: bool = True,
        model_name: str = "model",
) -> MLFlowDeploymentService:
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@step
def predictor(
        service: MLFlowDeploymentService,
        data: np.ndarray,
) -> np.ndarray:
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@step
def predictor(
        service: MLFlowDeploymentService,
        data: str,
) -> np.ndarray:
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
        min_accuracy: float = 0.9,
        workers: int = 1,
        timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    df = ingest_data(data_path=r"C:\Users\krish\PycharmProjects\python\MLops\olist_customers_dataset.csv")
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train(x_train, x_test, y_train, y_test)
    mse, rmse = model_evaluate(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=mse)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
