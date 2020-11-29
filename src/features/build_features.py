import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_filepath', default='data/interim/', type=click.Path(exists=True))
@click.argument('output_filepath', default='data/processed/', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn iterim data from (../interim) into
        features (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('reading interim data')
    df = pd.read_csv(input_filepath+'dataset.csv', low_memory=False)

    logger.info('handling missing data')
    df['missing_cat'] = 0
    df['missing_num'] = 0
    df['missing'] = 0
    df['missing_cat'] = df.isnull().sum(axis=1)
    df['missing_num'] = df.apply(lambda row: sum(row==-1) ,axis=1)
    df['missing'] = df['missing_cat']+df['missing_num']
    df = df[df.missing < 5]

    df[['tax', 'mpg']] = df[['tax', 'mpg']].fillna(-1)

    df.drop('reference', axis=1, inplace=True)

    # SPLIT INTO NUMERICAL AND CATEGORICAL DATASETS
    df_numerical = df[['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']]
    df_categorical = df[['brand', 'model', 'transmission', 'fuelType']]

    # ENCODE CATEGORICAL DATA
    logger.info('encoding categorical data')
    encoding = "Label Encoding"

    if encoding == "OneHot Encoding":
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        df_categorical = encoder.fit_transform(df_categorical)
        df_categorical = pd.DataFrame(df_categorical.toarray(), columns = encoder.get_feature_names(df_categorical.columns))  
    elif encoding == "Label Encoding":
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        df_categorical = df_categorical.apply(encoder.fit_transform)

    df_encoded = pd.concat([df_categorical, df_numerical], axis = 1)

    """
    # IMPUTE MISSING VALUES
    imputing = "simple"

    if imputing == "Iterative" :
        # MEMORY PROBLEMS
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=0)
    elif imputing == "Knn":
        # MEMORY PROBLEMS
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=2)
    else:
        import numpy as np
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    df_encoded = imputer.fit_transform(df_encoded)
    """
    
    df_encoded.to_csv(output_filepath+'dataset.csv', index=False)
    logger.info('data features saved')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
