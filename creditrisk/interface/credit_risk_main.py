import numpy as np
import pandas as pd
#import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from colorama import Fore, Style
from pathlib import Path
from dateutil.parser import parse

from creditrisk.params import *
from creditrisk.ml_logic.data import clean_data
from creditrisk.ml_logic.registry import save_model, save_results, load_model
from creditrisk.ml_logic.model import compile_model, initialize_model, train_model

def preprocess_and_train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    data_path = Path(LOCAL_DATA_PATH).joinpath("raw", "german.data")
    df = pd.read_csv(data_path, header=None, delimiter=' ')

    # Clean data using data.py
    # $CODE_BEGIN
    data = clean_data(data)
    # $CODE_END

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(['credit_risk'], axis=1))
    y = df['credit_risk'].values

    # Create (X_train, y_train, X_val, y_val) without data leaks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train model on the training set using `model.py`
    #Tune hyperparams
    params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    print('Best hyperparameters:', grid_search.best_params_)

    # Define the tuned hyperparameters
    max_depth = list(grid_search.best_params_.values())[0]
    n_estimators = list(grid_search.best_params_.values())[1]

    # Create a RandomForestClassifier object with the tuned hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # Retrain the model using the train set
    model.fit(X_train, y_train)


    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1-Score:', f1_score(y_test, y_pred))

    print("✅ preprocess_and_train() done")


if __name__ == '__main__':
    try:
        preprocess_and_train()
        # preprocess()
        # train()
        # pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
