# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle as p
import requests
import numpy as np
import json
import ast
import pandas as pd

pd.options.mode.chained_assignment = None


url = "http://0.0.0.0:5051/api/"


df = pd.read_csv("data/processed/dataset.csv", low_memory=False)
df_slice = df.loc[0:5]

X_slice = df_slice.drop("price", axis=1)

data_array = X_slice.to_numpy()
data_list = data_array.tolist()

r = requests.post(url, json={"data": data_list})
prediction_list = ast.literal_eval(r.content.decode("utf-8"))["predictions"]

df_slice["predictions"] = prediction_list
print(df_slice)
