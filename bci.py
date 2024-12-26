'''main script for processing pipeline including cross validation training and prediction'''

import argparse
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from param import get_param


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

def train(args:argparse.Namespace):
    param = get_param(args.experiment)
    pipeline = define_pipeline()




def define_pipeline() ->Pipeline:
    param_grid = {
        "pca__n_components": [40,65],
        "clf__alpha":[0.1,0.3,0.5],

    }
    clf = MLPClassifier()
    pipeline:Pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca',PCA()),
        ("clf", clf)
        ])


def main() -> None:
    '''main script'''
    args:argparse.Namespace = define_args()
    if args.mode == 'train':
        train(args)




if __name__ == '__main__':
    main()

