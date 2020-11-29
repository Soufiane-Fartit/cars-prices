#%% Import Packages
import pandas as pd

import os
#os.chdir('../..')
#print(os.getcwd())

#%% Read Data
df = pd.read_csv('data/interim/dataset.csv')

#%% Drop rows with missing values
df['missing_cat'] = 0
df['missing_num'] = 0
df['missing'] = 0
df['missing_cat'] = df.isnull().sum(axis=1)
df['missing_num'] = df.apply(lambda row: sum(row==-1) ,axis=1)
df['missing'] = df['missing_cat']+df['missing_num']
df = df[df.missing < 5]

#%% Drop columns with very few values
df.drop('reference', axis=1, inplace=True)

#%% Separate Categorical And Numerical Columns
df_numerical = df[['year', 'mileage', 'tax', 'mpg', 'engineSize', 'price']]
df_categorical = df[['brand', 'model', 'transmission', 'fuelType']]

#%% Encode Categorical Columns
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

#%% Impute Columns with missing values
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

#df_encoded = imputer.fit_transform(df_encoded)

#%% Save The Processed Data
df_encoded.to_csv('data/processed/dataset.csv')