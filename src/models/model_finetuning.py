# -*- coding: utf-8 -*-

""" This module is used to find the best hyperparameters combination to train the model """

import logging
from pathlib import Path
import os
import json
import click
from scipy.stats import uniform, truncnorm, randint
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import plotly.express as px
import plotly.io as pio
from utils_func import id_generator

pio.orca.config.executable = "/usr/local/bin/orca"


# pylint: disable=no-value-for-parameter
# pylint: disable=duplicate-code
@click.command()
@click.argument(
    "input_filepath", default="data/processed/", type=click.Path(exists=True)
)
@click.argument("output_filepath", default="models/", type=click.Path())
@click.argument("params_path", default="/models/parameters.json", type=click.Path())
def main(input_filepath, output_filepath, params_path):
    """Finds the best hyperparameters for the model
    and save them as in (parameters.json)
    """
    logger = logging.getLogger(__name__)

    # READING DATA
    logger.info("loading data")
    data = pd.read_csv(input_filepath + "dataset.csv", low_memory=False)
    features, targets = data.drop("price", axis=1), data["price"]

    # BUILDING ESTIMATOR
    logger.info("loading regressor and randomsearchcv")

    regr = RandomForestRegressor()

    # BUILDING RANDOMIZED SEARCH WITH CROSS-VALIDATION

    random_grid = {
        "n_estimators": randint(4, 200),
        "max_features": truncnorm(a=0, b=1, loc=0.25, scale=0.1),
        "max_depth": randint(20, 70),
        "min_samples_split": uniform(0.01, 0.199),
        "min_samples_leaf": randint(1, 5),
        "bootstrap": [True, False],
    }
    regr_random = RandomizedSearchCV(
        estimator=regr,
        param_distributions=random_grid,
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    # SEARCHING FOR BEST HYPERPARAMETERS
    search = regr_random.fit(features, targets)
    logger.info("best hyperparameters found")

    # MAKING A DIRECTORY FOR THE EXPERIMENT
    run_id = "run_" + id_generator()
    if not os.path.exists(output_filepath + "hyperparams-search/" + run_id):
        os.makedirs(output_filepath + "hyperparams-search/" + run_id)

    # READING THE RESULTS OF THE RANDOMIZED SEARCH
    cv_result = search.cv_results_
    cv_result.pop("params")
    cv_result = pd.DataFrame(cv_result)

    # SAVE THE RANDOMIZED SEARCH RESULTS IN THE EXPERIMENT DIRECTORY
    cv_result.to_csv(
        output_filepath + "hyperparams-search/" + run_id + "/result.csv", index=False
    )

    # SAVING A PARALLEL COORDINATES PLOT OF THE HYPERPARAMETERS IN THE EXPERIMENT DIRECTORY
    fig = px.parallel_coordinates(cv_result, color="mean_test_score")
    fig.write_image(
        output_filepath
        + "hyperparams-search/"
        + run_id
        + "/parallelcoordinatesplot.png",
        height=600,
        width=1100,
    )

    # SAVING THE BEST HYPERPARAMETERS IN THE EXPERIMENT DIRECTORY TO KEEP TRACK
    with open(
        output_filepath + "hyperparams-search/" + run_id + "/best_parameters.json", "w"
    ) as outfile:
        json.dump(search.best_params_, outfile)

    # SAVING THE BEST HYPERPARAMETERS TO BE USED IN THE FUTURE
    with open(str(project_dir) + "/" + params_path, "w") as outfile:
        json.dump(search.best_params_, outfile)
    logger.info("saved model parameters")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
