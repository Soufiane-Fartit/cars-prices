# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle
import numpy as np
import json


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    data_list = data['data']
    data_array = np.array(data_list)
    #prediction = np.array2string(model.predict(data_array))
    prediction = model.predict(data_array)
    pred_list = prediction.tolist()

    return {'predictions':pred_list}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    model_id = '6j0897'
    modelfile = str(project_dir)+'/models/model_'+model_id+'.pkl'
    model = pickle.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0', port=5051)
