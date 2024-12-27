'''main script for processing pipeline including cross validation training and prediction'''

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from param import get_param,get_prefix, PREPROCESSED_PATH, RANDOM_STATE, TEST_SIZE


def define_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description='''Main script for processing pipeline including cross validation training and prediction''')
    subparser = parser.add_subparsers(dest='mode')
    subparser_train = subparser.add_parser("train")
    subparser_train.add_argument("--experiment", type=int, default=1, choices=[1,2,3,4,5,6], help="experiment no. (1 to 6) Default = 1")
    subparser_train.add_argument("--subject", type=int, default=1, help="subject no. (1 to 109). Default = 1")
    subparser_predict = subparser.add_parser("predict")
    subparser_predict.add_argument("--experiment", type=int, default=1, choices=[1,2,3,4,5,6], help="experiment no. (1 to 6) Default = 1")
    subparser_predict.add_argument("--subject", type=int, default=1, help="subject no. (1 to 109). Default = 1")
    return parser.parse_args()

def split_data(df:pd.DataFrame, label:list)->tuple:
    '''train test split'''
    X = df[(df['condition'] == label[0]) | (df['condition'] == label[1])]
    y = pd.Series(X['condition'].values)
    X = X.drop(["condition", "Unnamed: 0", "epoch"], axis=1)

    # Use stratified to prevent training bias
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    return (X_train, y_train, X_test, y_test)
    

def define_pipeline() ->Pipeline:
    param_grid:dict = {
        "pca__n_components": [40,65],
        "clf__alpha":[0.3,0.5],
    }
    clf = MLPClassifier(max_iter=1000)
    pipeline:Pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca',PCA()),
        ("clf", clf)
        ])

def train(args:argparse.Namespace)-> None:
    '''train using pipeline using args parameter'''
    param:dict = get_param(args.experiment)

    # reading preprocessed file as per args params
    prefix:str = get_prefix(args.subject) 
    try:
        df:pd.DataFrame = pd.read_csv(f"{PREPROCESSED_PATH}S{prefix}{args.subject}E{args.experiment}.csv")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    X_train, y_train = split_data(df, param['label'])
    pipeline = define_pipeline()


def predict(args:argparse.Namespace) -> None:
    '''predict using pipeline using args parameter'''
    pass

def train_all(args:argparse.Namespace) -> None:
    '''train_all using pipeline'''
    pass

def main():
    '''main script'''
    args:argparse.Namespace = define_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        train_all(args)




if __name__ == '__main__':
    main()

