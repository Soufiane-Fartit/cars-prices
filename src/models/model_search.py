# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd


@click.command()
@click.argument('input_filepath', default='data/processed/', type=click.Path(exists=True))
@click.argument('output_filepath', default='models/', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('training model')


    df = pd.read_csv(input_filepath+'dataset.csv', low_memory=False)

    X, y = df.drop('price',axis=1), df['price']
    print(X.head())
    print(y.head())
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X_train, y_train)
