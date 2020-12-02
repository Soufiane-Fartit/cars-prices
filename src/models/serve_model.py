# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from flask import Flask, request
import pickle
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", default=5000,
                    help="port of the api")
parser.add_argument("-m", "--model_id", default="fbgjqx",
                    help="port of the api")
args = parser.parse_args()
port = args.port
model_id = args.model_id

app = Flask(__name__)


@app.route("/api/", methods=["POST"])
def makecalc():
    data = request.get_json()
    data_list = data["data"]
    data_array = np.array(data_list)
    prediction = model.predict(data_array)
    pred_list = prediction.tolist()

    return {"predictions": pred_list}


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    modelfile = (
        str(project_dir) + "/models/models-training/run_" + model_id + "/model.pkl"
    )

    print(f'Using model : {model_id}')

    model = pickle.load(open(modelfile, "rb"))
    app.run(debug=True, host="0.0.0.0", port=port)
