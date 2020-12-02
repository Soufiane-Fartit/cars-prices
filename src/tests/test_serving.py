# -*- coding: utf-8 -*-
import requests
import ast
import pandas as pd
from sklearn.metrics import r2_score
import argparse

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", default=5000,
                    help="port of the api")
args = parser.parse_args()
port = args.port

url = "http://0.0.0.0:"+port+"/api/"


df = pd.read_csv("data/processed/dataset.csv", low_memory=False)
df_slice = df  # .loc[0:5]

X_slice = df_slice.drop("price", axis=1)

data_array = X_slice.to_numpy()
data_list = data_array.tolist()

r = requests.post(url, json={"data": data_list})
assert r.status_code == 200

prediction_list = ast.literal_eval(r.content.decode("utf-8"))["predictions"]
df_slice["predictions"] = prediction_list
# print(df_slice)

assert len(prediction_list) == df.shape[0]
assert r2_score(df['price'], df['predictions']) > 0.7
