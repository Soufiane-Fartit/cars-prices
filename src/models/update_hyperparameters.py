# -*- coding: utf-8 -*-

""" This module updates the models/parameters.json file """

import logging
import glob
from pathlib import Path
import os
import json
import click
import numpy as np
import pandas as pd

# pylint: disable=no-value-for-parameter
# pylint: disable=duplicate-code
@click.command()
@click.argument("params_path", default="/models/parameters.json", type=click.Path())
def main(params_path):
    """SEARCH THROUGH HYSTORY OF HYPERPARAMETERS SEARCHING AND SET THE BEST
        COMBINATION OF HYPERPARAMETERS TO BE USED

    Args:
        params_path (str): path to the best set of hyperparameters
    """
    logger = logging.getLogger(__name__)

    all_files = [x[0]+'/result.csv' for x in os.walk('models/hyperparams-search')][1:]
    list_of_dfs = []
    for filename in all_files:
        dataframe = pd.read_csv(filename)
        list_of_dfs.append(dataframe)
    concat = pd.concat(list_of_dfs, axis=0, ignore_index=True)

    # FILTER HYPERPARAMETERS BY RESULT
    #    BY SPEED
    concat = concat[concat['mean_fit_time']<20]
    concat = concat[concat['std_fit_time']<5]
    concat = concat[concat['mean_score_time']<0.67]
    concat = concat[concat['std_score_time']<0.04]
    #    BY PERFORMANCE
    concat = concat[concat['mean_test_score']>0.67]
    concat = concat[concat['std_test_score']<0.04]
    #    BY DATE
    # TO BE DONE ?

    assert concat.shape[0] > 0 , 'no hyperparams combination passed the tests'

    # RETAIN THE COMBINATION WITH HIGHEST SCORE
    best_combination = concat.sort_values('mean_test_score', ascending=False).iloc[0].to_dict()
    
    try :
        # READ OLD HYPERPARAMETERS TO COMPARE
        with open(str(project_dir) + "/" + params_path, "r") as infile:
            old_params = json.load(infile)
    except :
        pass

    # IF THERE EXISTS A PREVIOUS VERSION COMPARE, OTHERWISE CONTINUE
    if 'old_params' in locals():
        assert best_combination['mean_test_score'] > old_params['mean_test_score'], 'old model is better'


    # CONVERT IF TYPE ERROR RAISED
    def convert(o):
        if isinstance(o, np.generic): return o.item()  
        raise TypeError

    with open(str(project_dir) + "/" + params_path, "w") as outfile:
        json.dump(best_combination, outfile, default=convert)
    logger.info("saved best combination of parameters")

if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()