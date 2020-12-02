# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
import json
import pandas as pd
from utils_func import update_history, id_generator, save_model


@click.command()
@click.argument(
    "input_filepath", default="data/processed/", type=click.Path(exists=True)
)
@click.argument("output_filepath", default="models/", type=click.Path())
@click.argument("params_path", default="/models/parameters.json", type=click.Path())
def main(input_filepath, output_filepath, params_path):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    """

    logger = logging.getLogger(__name__)

    # LOADING MODEL PARAMETERS
    with open(str(project_dir) + params_path, "r") as f:
        params = json.load(f)
    logger.info("loaded model parameters")
    print(params)

    # LOADING DATA FEATURES
    logger.info("loading data features")
    df = pd.read_csv(input_filepath + "dataset.csv", low_memory=False)
    X, y = df.drop("price", axis=1), df["price"]

    # TRAINING MODEL
    logger.info("training model")
    from sklearn.ensemble import RandomForestRegressor

    regr = RandomForestRegressor(**params)
    regr.fit(X, y)

    # SAVING MODEL
    logger.info("saving model")
    model_id = id_generator()
    model_name = "model_" + model_id

    # MAKING A DIRECTORY FOR THE TRAINING RUN
    if not os.path.exists(output_filepath + "models-training/run_" + model_id):
        os.makedirs(output_filepath + "models-training/run_" + model_id)

        # SAVING THE MODEL IN THE RUN DIRECTORY
    save_model(output_filepath + "models-training/run_" + model_id + "/model.pkl", regr)

    # SAVING THE HYPERPARAMETERS USED TO TRAIN THE MODEL IN THE RUN DIRECTORY
    with open(
        output_filepath + "models-training/run_" + model_id + "/params.json", "w"
    ) as f:
        json.dump(params, f)

    logger.info("model saved")

    # LOGGING MODEL INTO HISTORY FILE
    update_history(
        output_filepath + "models-training/models_history.json",
        model_id,
        model_name,
        regr,
        params,
    )
    logger.info("saved model metadata")

    return 0


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
