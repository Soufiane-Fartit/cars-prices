# -*- coding: utf-8 -*-

""" This module is used to deploy an API to request predictions from the model """

import logging
from pathlib import Path
import json
import pickle
from flask import Flask, request
import numpy as np


app = Flask(__name__)


@app.route("/api/", methods=["POST"])
def makecalc():
    """USES THE MODEL TO MAKE PREDICTIONS ON THE DATA SENT THROUGH THE API

    Returns:
        list: list of the predictions
    """
    data = request.get_json()
    data_list = data["data"]
    data_array = np.array(data_list)
    prediction = model.predict(data_array)
    pred_list = prediction.tolist()

    return {"predictions": pred_list}


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
    port = params["port"]

    modelfile = (
        str(project_dir) + "/models/models-training/run_" + model_id + "/model.pkl"
    )

    print(f"Using model : {model_id}")

    model = pickle.load(open(modelfile, "rb"))
    app.run(debug=True, host="0.0.0.0", port=port)
