# -*- coding: utf-8 -*-

""" This module serves to validate a model on the dataset """

import logging
from pathlib import Path
import json
import pickle
import pandas as pd


def main(model_id, input_filepath="data/processed/"):

    """

    Validate the trained model

    """

    logger = logging.getLogger(__name__)

    logger.info(f"Validating model : {model_id}")

    logger.info(f"loading model : {model_id}")
    modelfile = (
        str(project_dir) + "/models/models-training/run_" + model_id + "/model.pkl"
    )
    model = pickle.load(open(modelfile, "rb"))

    logger.info("loading dataset")
    data = pd.read_csv(input_filepath + "dataset.csv", low_memory=False)

    logger.info("making predictions")
    data["predictions"] = model.predict(data.drop("price", axis=1))
    print(data.head())

    # EVALUATE MODELS' CONSISTENCY ON DIFFERENT BRANDS
    corrs = data.groupby("brand")[["price", "predictions"]].corr().iloc[0::2, -1]
    assert corrs.isna().sum() == 0, "model unstable, NAN values"
    assert (corrs > 0.7).sum() == len(
        corrs
    ), "model gives bad predictions for some classes"
    assert corrs.mean() > 0.8, "model overall bad"
    assert corrs.var() < 0.3, "model have high variance"

    # EVALUATE MODELS' CONSISTENCY ON DIFFERENT BRANDS
    corrs = data.groupby("model")[["price", "predictions"]].corr().iloc[0::2, -1]
    assert corrs.isna().sum() == 0, "model unstable, NAN values"
    assert (corrs > 0.7).sum() == len(
        corrs
    ), "model gives bad predictions for some classes"
    assert corrs.mean() > 0.8, "model overall bad"
    assert corrs.var() < 0.3, "model have high variance"


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    with open(str(project_dir) + "/models/deployment.json", "r") as infile:
        params = json.load(infile)

    model_id = params["model_id"]

    main(model_id)
