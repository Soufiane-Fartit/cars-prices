# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pickle
import json
import pandas as pd



@click.command()
@click.argument('input_filepath', default='data/processed/', type=click.Path(exists=True))
@click.argument('output_filepath', default='models/', type=click.Path())
@click.argument('params_path', default='/src/models/parameters.json', type=click.Path())
def main(input_filepath, output_filepath, params_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)

    with open(str(project_dir)+params_path, 'r') as f:
        params = json.load(f)
    logger.info('loaded model parameters')
    print(params)

    logger.info('loading data features')
    df = pd.read_csv(input_filepath+'dataset.csv', low_memory=False)
    X, y = df.drop('price',axis=1), df['price']

    logger.info('training model')
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(**params)
    regr.fit(X, y)

    logger.info('saving model')
    pkl_filename = output_filepath+"pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(regr, file)
    logger.info('model saved')

    return 0

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    
    main()