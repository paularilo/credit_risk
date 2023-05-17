import pandas as pd
from sklearn.preprocessing import LabelEncoder

#from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from creditrisk.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    # Set the column names
    df.columns = COLUMN_NAMES

    # Convert categorical variables to numerical variables
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])

    # Compress raw_data by setting types to DTYPES_RAW
    #df = df.astype(DTYPES_RAW)

    # Remove buggy data
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)

    print("✅ data cleaned")

    return df
