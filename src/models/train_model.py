# -*- coding: utf-8 -*-

""" This module is used to train a model on a combination of hyperparameters and data """

from pathlib import Path
import logging
import os
import json
import click
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from utils_func import (
    update_history,
    id_generator,
    save_model,
    generate_features_importance_plot,
    plot_trees,
)

# pylint: disable=no-value-for-parameter
# pylint: disable=duplicate-code
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
    with open(str(project_dir) + "/" + params_path, "r") as infile:
        params = json.load(infile)
    logger.info("loaded model parameters")
    print(params)

    # LOADING DATA FEATURES
    logger.info("loading data features")
    data = pd.read_csv(input_filepath + "dataset.csv", low_memory=False)
    features, targets = data.drop("price", axis=1), data["price"]

    # TRAINING MODEL
    logger.info("training model")
    regr = RandomForestRegressor(**params)
    regr.fit(features, targets)

    # SAVING MODEL
    logger.info("saving model")
    model_id = id_generator()
    model_name = "model_" + model_id

    # MAKING A DIRECTORY FOR THE TRAINING RUN
    if not os.path.exists(output_filepath + "models-training/run_" + model_id):
        os.makedirs(output_filepath + "models-training/run_" + model_id)

        # SAVING THE MODEL IN THE RUN DIRECTORY
    save_model(output_filepath + "models-training/run_" + model_id + "/model.pkl", regr)

    # SAVING FEATURES IMPORTANCE PLOT
    generate_features_importance_plot(regr, features, model_id)

    # SAVING TREES PLOT
    plot_trees(regr, features.columns, "price", model_id)

    # SAVING THE HYPERPARAMETERS USED TO TRAIN THE MODEL IN THE RUN DIRECTORY
    with open(
        output_filepath + "models-training/run_" + model_id + "/params.json", "w"
    ) as outfile:
        json.dump(params, outfile)

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

    # SAVE MODEL ID
    with open(str(project_dir) + "/" + "models/deployment.json", "w") as outfile:
        json.dump({'model_id': model_id, 'port': 33507}, outfile)
    logger.info("saved model id for deployment")

    return 0


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
