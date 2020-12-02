# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import json
import click
import pandas as pd

"""
#Example on R_Square and Adjusted R Square
import statsmodels.api as sm
X_addC = sm.add_constant(X)
result = sm.OLS(Y, X_addC).fit()
print(result.rsquared, result.rsquared_adj)

#MSE
from sklearn.metrics import mean_squared_error
import math
print(mean_squared_error(Y_test, Y_predicted))
print(math.sqrt(mean_squared_error(Y_test, Y_predicted)))

#MAE
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test, Y_predicted))


# model good for 1 country and bad for another country
=> don't validate

"""

# pylint: disable=no-value-for-parameter
# pylint: disable=duplicate-code
@click.command()
@click.argument(
    "input_filepath", default="data/processed/", type=click.Path(exists=True)
)
@click.argument("output_filepath", default="models/", type=click.Path())
@click.argument("params_path", default="/src/models/parameters.json", type=click.Path())
def main(input_filepath, output_filepath, params_path):

    """

    Measures metrics to evaluate the trained model

    """

    logger = logging.getLogger(__name__)

    logger.info("evaluating model")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
