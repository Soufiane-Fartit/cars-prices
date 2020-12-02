# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

# from dotenv import find_dotenv, load_dotenv
import glob
import pandas as pd


# pylint: disable=no-value-for-parameter
@click.command()
@click.argument("input_filepath", default="data/raw/", type=click.Path(exists=True))
@click.argument("output_filepath", default="data/interim/", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info("loading brut data")

    list_of_dfs = []
    all_files = glob.glob(input_filepath + "*.csv")

    for filename in all_files:
        brand = filename[len(input_filepath) : -4]
        df = pd.read_csv(filename, index_col=None, header=0)

        df.insert(0, "brand", brand)

        # parse price
        df["price"] = df["price"].replace("[\£,]", "", regex=True).astype(float)
        df["price"] = df["price"].fillna(-1).astype(int)

        # clean mileage
        if "mileage2" in df.columns:
            df[["mileage"]] = (
                df[["mileage", "mileage2"]]
                .fillna(-1)
                .replace("Unknown", "-1")
                .replace(",", "", regex=True)
                .astype(float)
                .astype(int)
                .max(axis=1)
            )
            df.drop("mileage2", axis=1, inplace=True)

        # clean engine size
        if "engine size2" in df.columns:
            es = pd.to_numeric(df["engine size"], errors="coerce").fillna(-1)
            es2 = pd.to_numeric(df["engine size2"], errors="coerce").fillna(-1)
            df["engine size"] = pd.DataFrame([es, es2]).max(axis=0)
            df.drop("engine size2", axis=1, inplace=True)

        # clean fluel type
        if "fuel type2" in df.columns:

            def func(x):
                if x.values[0] is None:
                    return None
                else:
                    return df.loc[x.name, x.values[0]]

            df[["fuel type"]] = pd.DataFrame(
                df[["fuel type", "fuel type2"]].apply(
                    lambda x: x.first_valid_index(), axis=1
                )
            ).apply(func, axis=1)
            df.drop("fuel type2", axis=1, inplace=True)

        df["year"] = df["year"].fillna(-1).astype(int)

        df = df.rename(
            columns={
                "engine size": "engineSize",
                "tax(£)": "tax",
                "fuel type": "fuelType",
            }
        )

        list_of_dfs.append(df)

    logger.info("all files have been processed")

    concat = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    concat.to_csv(output_filepath + "dataset.csv", index=False)

    logger.info("interim data saved")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
