# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import glob
import pandas as pd


@click.command()
@click.argument('input_filepath', default='data/raw/', type=click.Path(exists=True))
@click.argument('output_filepath', default='data/interim/', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    list_of_dfs = []
    all_files = glob.glob(input_filepath + "*.csv")

    for filename in all_files:
        brand = filename[len(input_filepath):-4]
        df = pd.read_csv(filename, index_col=None, header=0)
        #df['brand'] = brand
        df.insert(0, 'brand', brand)
        list_of_dfs.append(df)
    
    concat = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    concat.to_csv(output_filepath + 'dataset.csv')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
