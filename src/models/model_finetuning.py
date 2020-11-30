# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import json
from scipy.stats import uniform, truncnorm, randint
import pandas as pd


@click.command()
@click.argument('input_filepath', default='data/processed/', type=click.Path(exists=True))
@click.argument('output_filepath', default='models/', type=click.Path())
@click.argument('params_path', default='/src/models/parameters.json', type=click.Path())
def main(input_filepath, output_filepath, params_path):
    """ Finds the best hyperparameters for the model
        and save them as in (parameters.json)
    """
    logger = logging.getLogger(__name__)

    logger.info('loading data')
    df = pd.read_csv(input_filepath+'dataset.csv', low_memory=False)
    X, y = df.drop('price',axis=1), df['price']

    logger.info('loading regressor and randomsearchcv')
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor()

    from sklearn.model_selection import RandomizedSearchCV
    random_grid = {'n_estimators': randint(4,200),
               'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
               'max_depth': randint(20,70),
               'min_samples_split': uniform(0.01, 0.199),
               'min_samples_leaf': randint(1,5),
               'bootstrap': [True, False]}

    regr_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    search = regr_random.fit(X, y)
    logger.info('best hyperparameters found')
    print(search.best_params_)

    with open(str(project_dir)+params_path, 'w') as f:
        json.dump(search.best_params_, f)
    logger.info('saved model parameters')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    
    main()