'''main script for processing pipeline including cross validation training and prediction'''

import argparse
import os
import shutil
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from param import get_param,get_prefix, PREPROCESSED_PATH, RANDOM_STATE, TEST_SIZE, MEMORY_CACHE_PATH, MODEL_PATH


def define_args()->argparse.Namespace:
    '''define arguments in command line'''
    parser = argparse.ArgumentParser(description='Main script for processing pipeline including cross validation training and prediction', formatter_class=argparse.RawTextHelpFormatter)
    subparser = parser.add_subparsers(dest='mode')
    subparser_train = subparser.add_parser("train", formatter_class=argparse.RawTextHelpFormatter)
    subparser_train.add_argument("--experiment", type=int, default=1, choices=[1,2,3,4,5,6], help=(
            "experiment 1: left first vs right fist\n"
            "experiment 2: imagine left fist vs right fist\n"
            "experiment 3: both fists vs both feet\n"
            "experiment 4: imagine both fists vs both feet\n"
            "experiment 5: rest vs left fist\n"
            "experiment 6: rest vs imagine both feet\n"
            "Default=1"
        ))
    subparser_train.add_argument("--subject", type=int, default=1, help="subject no. (1 to 109). Default = 1")
    subparser_train.add_argument("--verbose", action='store_true', help="Display additional info on training in terminal")
    subparser_predict = subparser.add_parser("predict", formatter_class=argparse.RawTextHelpFormatter )
    subparser_predict.add_argument("--experiment", type=int, default=1, choices=[1,2,3,4,5,6], help=(
            "experiment 1: left first vs right fist\n"
            "experiment 2: imagine left fist vs right fist\n"
            "experiment 3: both fists vs both feet\n"
            "experiment 4: imagine both fists vs both feet\n"
            "experiment 5: rest vs left fist\n"
            "experiment 6: rest vs imagine both feet\n"
            "Default=1"
        )
    )
    subparser_predict.add_argument("--subject", type=int, default=1, help="subject no. (1 to 109). Default = 1")
    subparser_clear = subparser.add_parser("clear")
    return parser.parse_args()

def split_data(df:pd.DataFrame, label:list)->tuple:
    '''train test split'''
    df=df.sample(frac=1, random_state=42)
    X = df[(df['condition'] == label[0]) | (df['condition'] == label[1])]
    y = pd.Series(X['condition'].values)
    X = X.drop(["condition", "Unnamed: 0", "epoch"], axis=1)

    # Use stratified to prevent training bias
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30, random_state=RANDOM_STATE)
    return (X_train, X_test, y_train, y_test, X, y )

def define_grid()->dict:
    '''return param_grid for RandomizedSearchCV'''
    return {
        "pca__n_components": [40,65],
        "clf__alpha":[0.1,0.3],
    }
 
def define_pipeline(args:argparse.Namespace) ->Pipeline:
    '''define and return Pipeline Object for RandomizedSearchCV'''
    try:
        os.mkdir(MEMORY_CACHE_PATH)
    except FileExistsError:
        pass
    clf = MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca',PCA()),
        ("clf", clf)
        ], memory=MEMORY_CACHE_PATH, verbose=(True if args.verbose else False))

def train(args:argparse.Namespace)-> None:
    '''train using pipeline using args parameter'''
    try:
        os.mkdir(MODEL_PATH)
    except FileExistsError:
        pass

    param:dict = get_param(args.experiment)
    prefix:str = get_prefix(args.subject)
    # reading preprocessed file as per args params
    try:
        df:pd.DataFrame = pd.read_csv(f"{PREPROCESSED_PATH}S{prefix}{args.subject}E{args.experiment}.csv")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    X_train, _, y_train, _ = split_data(df, param['label'])
    cv:int = 5
    # combining gridsearch of best param together with cross fold val. Hence the amount of fitting is no. of combination
    # of parameters x Kfold. As below it's params(2x2) x cv(5) = 20. Hence train model 10 times. 
    grid = GridSearchCV(define_pipeline(args), define_grid(), n_jobs=-1, cv=cv, verbose=(4 if args.verbose else 0))
    grid.fit(X_train, y_train)

    # training results included in grid.cv_results_
    result = grid.cv_results_
    # parsing cross validation score across the folds. Use index of best params combination - rank 1 in key('rank_test_score') 
    # and use that to get score of each fold -> key(ie 'split0', 'split1').  
    split:list = []
    for i in range(cv):
        split.append(f"split{i}")
    cv_score:list = [round(float(result[value + '_test_score'][result['rank_test_score'].tolist().index(1)]),4) for value in split]
    # print score of each fold
    print(cv_score)
    # print cross_val_score
    print(f"cross_val_score: {np.array(cv_score).mean():.4f}")
    # save estimator
    filename:str = f"S{prefix}{args.subject}E{args.experiment}.pkl" 
    print(f"saving estimator {filename} in {MODEL_PATH}")
    with open(f"{MODEL_PATH}{filename}", "wb") as file:
        pickle.dump(grid, file)
    


def predict(args:argparse.Namespace) -> None:
    '''predict using existing pre trained model and streaming test data'''
    prefix:str = get_prefix(args.subject)
    filePATH:str = f"{MODEL_PATH}S{prefix}{args.subject}E{args.experiment}.pkl"
    
    if os.path.exists(filePATH):
        with open(filePATH,"rb") as file:
            grid = pickle.load(file)
    else:
        print("Model does not exist. Please train specific model first")
        return
    param:dict = get_param(args.experiment)
    prefix:str = get_prefix(args.subject)
    # reading preprocessed file as per args params
    try:
        df:pd.DataFrame = pd.read_csv(f"{PREPROCESSED_PATH}S{prefix}{args.subject}E{args.experiment}.csv")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    X_train, X_test, y_train, y_test, X, y = split_data(df, param['label'])

    print("predict", grid.predict(X))
    T1 = 0
    T2 = 0
    for i in grid.predict(X):
        if i == "T1":
            T1+=1
        else:
            T2+=1
    print("t1", T1)
    print("t2", T2)
     
    print("score", grid.score(X, y))
    


def train_all(args:argparse.Namespace) -> None:
    '''train_all using pipeline'''
    pass

def main():
    '''main script for 1) train, 2) predict and 3) train all by subject and experiments'''
    args:argparse.Namespace = define_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    elif args.mode == 'clear':
        try:
            shutil.rmtree(MEMORY_CACHE_PATH)
            print("memory cache cleared")
        except Exception as e:
            print(f"Error occured :{e}")
            return
    else:
        train_all(args)




if __name__ == '__main__':
    main()

