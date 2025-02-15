# -*- coding: utf-8 -*-

""" This module process the dataset to create usefull features """

import logging
from pathlib import Path
import click

# from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# pylint: disable=no-value-for-parameter
# pylint: disable=duplicate-code
@click.command()
@click.argument("input_filepath", default="data/interim/", type=click.Path(exists=True))
@click.argument("output_filepath", default="data/processed/", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn iterim data from (../interim) into
    features (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("reading interim data")
    data = pd.read_csv(input_filepath + "dataset.csv", low_memory=False)

    logger.info("handling missing data")
    data["missing_cat"] = 0
    data["missing_num"] = 0
    data["missing"] = 0
    data["missing_cat"] = data.isnull().sum(axis=1)
    data["missing_num"] = data.apply(lambda row: sum(row == -1), axis=1)
    data["missing"] = data["missing_cat"] + data["missing_num"]
    data = data[data.missing < 5]

    data[["tax", "mpg"]] = data[["tax", "mpg"]].fillna(-1)

    data.drop("reference", axis=1, inplace=True)

    # SPLIT INTO NUMERICAL AND CATEGORICAL DATASETS
    df_numerical = data[["year", "mileage", "tax", "mpg", "engineSize", "price"]]
    df_categorical = data[["brand", "model", "transmission", "fuelType"]]

    # ENCODE CATEGORICAL DATA
    logger.info("encoding categorical data")
    encoding = "Label Encoding"

    if encoding == "OneHot Encoding":
        encoder = OneHotEncoder(handle_unknown="ignore")
        df_categorical = encoder.fit_transform(df_categorical)
        df_categorical = pd.DataFrame(
            df_categorical.toarray(),
            columns=encoder.get_feature_names(df_categorical.columns),
        )
    elif encoding == "Label Encoding":
        encoder = LabelEncoder()
        df_categorical = df_categorical.apply(encoder.fit_transform)

    df_encoded = pd.concat([df_categorical, df_numerical], axis=1)

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

    df_encoded.to_csv(output_filepath + "dataset.csv", index=False)
    logger.info("data features saved")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
