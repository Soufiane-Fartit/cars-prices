# -*- coding: utf-8 -*-

""" This module offers util functions to be called and used
    in other modules
"""

from datetime import datetime
import json
import pickle
import string
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import tree


def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    """GENERATE A RANDOM STRING TO BE USED AS AN ID

    Args:
        size (int, optional): size of the string. Defaults to 6.
        chars (str, optional): charachters to be used to generate the string.
                            Defaults to string.ascii_lowercase+string.digits.

    Returns:
        [str]: a random chain of charachters
    """
    return "".join(random.choice(chars) for _ in range(size))


def save_model(path, model):
    """SAVE MODEL INTO PICKLE FILE

    Args:
        path (str): path where to save the model
        model (binary): the model to be saved
    """

    with open(path, "wb") as file:
        pickle.dump(model, file)


def update_history(models_hist_path, model_id, model_name, model, params):
    """SAVE METADATA RELATED TO THE TRAINED MODEL INTO THE HISTORY FILE

    Args:
        models_hist_path (str): path to the history file
        model_id (str): unique id of the model
        model_name (str): model name = "model_"+model_id+".pkl"
        model (binary): binary file of the model
        params (dict): dictionnary containing the hyper-parameters
                        used to fit the model
    """

    model_metadata = dict()
    model_metadata["trained"] = str(datetime.now())
    model_metadata["model_type"] = type(model).__name__
    model_metadata["model_id"] = model_id
    model_metadata["params"] = params
    print(model_metadata)

    with open(models_hist_path, "r+") as outfile:
        try:
            hist = json.load(outfile)
            hist[model_name] = model_metadata
            outfile.seek(0)
            json.dump(hist, outfile, indent=4)
        except json.decoder.JSONDecodeError:
            json.dump({model_name: model_metadata}, outfile, indent=4)


def update_history_add_eval(
    models_hist_path, model_id=None, model_name=None, metrics=None
):
    """ADD EVALUATION METRICS THE HISTORY FILE FOR THE SPECIFIED MODEL

    Args:
        models_hist_path (str): path to the history file
        model_id (str, optional): the id of the model. Defaults to None.
        model_name (str, optional): the name of the model. Defaults to None.
        metrics (dict, optional): a dictionnary containing metadata related
                                    to the model evaluation. Defaults to None.
    """

    assert (
        model_id is not None or model_name is not None
    ), "At least the model id or name must be given"
    assert models_hist_path is not None, "You must specify the path to the history file"

    if not model_name:
        model_name = "model_" + model_id + ".pkl"

    eval_metadata = dict()
    eval_metadata["datetime"] = str(datetime.now())
    eval_metadata["metrics"] = metrics

    with open(models_hist_path, "r+") as outfile:
        try:
            hist = json.load(outfile)
            hist[model_name]["evaluation"] = eval_metadata
            outfile.seek(0)
            json.dump(hist, outfile, indent=4)
        except json.decoder.JSONDecodeError:
            print("cannot save evaluation metadata")


def generate_features_importance_plot(model, features, model_id):
    """GENERATES A PLOT DESCRIBING FEATURES IMPORTANCE FOR THE MODEL
    TO MAKE THE PREDICTION.

    Args:
        model (tree-based model): a tree based model (decision tree, random forest ...)
        features (pandas dataframe): a table of the features on which we trained the model
        model_id (str): the unique id of the model
    """
    mean_importances = model.feature_importances_
    importances_indices = np.argsort(mean_importances)[::-1]
    ordered_columns = [features.columns[i] for i in importances_indices]
    importances = pd.DataFrame(
        [tree.feature_importances_ for tree in model.estimators_],
        columns=features.columns,
    )
    importances = importances[ordered_columns]
    _, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x="variable", y="value", ax=ax, data=pd.melt(importances))
    figure = ax.get_figure()
    figure.savefig(
        "models/models-training/run_" + model_id + "/features_importance.png"
    )


def plot_trees(rf, feature_names, target_names, model_id):
    """GENERATES A PLOT THAT SHOWS THE DECISION MAKING OF THE TREES

    Args:
        rf (model): a tree based model (random forest ...)
        feature_names (list): names of the columns of the training set
        target_names (str): name of the target columns
        model_id (str): unique id of the model
    """
    fn = feature_names
    cn = target_names
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
    for index in range(0, 5):
        tree.plot_tree(
            rf.estimators_[index],
            feature_names=fn,
            class_names=cn,
            filled=True,
            ax=axes[index],
        )

        axes[index].set_title("Estimator: " + str(index), fontsize=11)
    fig.savefig("models/models-training/run_" + model_id + "/Trees.png")
