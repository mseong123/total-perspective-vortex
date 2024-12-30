'''main script for processing pipeline including cross validation training and prediction'''

import argparse
import os
import shutil
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from param import get_param,get_prefix, PREPROCESSED_PATH, \
    RANDOM_STATE, TEST_SIZE, MEMORY_CACHE_PATH, MODEL_PATH, \
    BATCH_SIZE



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
    parser.add_argument("--verbose", action='store_true', help="Display additional info on training in terminal")
    return parser.parse_args()

def split_data(df:pd.DataFrame, label:list)->tuple:
    '''train test split and process dataframe'''
    X = df[(df['condition'] == label[0]) | (df['condition'] == label[1])]
    y = X['condition']
    X = X.drop(["condition", "Unnamed: 0", "epoch"], axis=1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30,random_state=RANDOM_STATE)

    return (X_train, X_test, y_train, y_test)


def define_grid()->dict:
    '''return param_grid for RandomizedSearchCV'''
    return {
        "pca__n_components": [45, 65],
        "clf__alpha":[0.1, 0.3],
    }
 
def define_pipeline(args:argparse.Namespace) ->Pipeline:
    '''define and return Pipeline Object for RandomizedSearchCV'''
    try:
        os.mkdir(MEMORY_CACHE_PATH)
    except FileExistsError:
        pass
    clf = MLPClassifier(max_iter=1000,hidden_layer_sizes=(40,),random_state=RANDOM_STATE, early_stopping=True, verbose=(True if args.verbose else False))
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca',PCA()),
        ("clf", clf)
        ], memory=MEMORY_CACHE_PATH, verbose=(True if args.verbose else False))

def train(experiment:int, subject:int, args:argparse.Namespace, train_all:bool, indiv_score:list | None)-> None:
    '''train using pipeline using args parameter'''
    try:
        os.mkdir(MODEL_PATH)
    except FileExistsError:
        pass

    param:dict = get_param(experiment)
    prefix:str = get_prefix(subject)
    # reading preprocessed file as per args params
    try:
        df:pd.DataFrame = pd.read_csv(f"{PREPROCESSED_PATH}S{prefix}{subject}E{experiment}.csv")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("exiting..")
        exit()
    X_train, X_test, y_train, y_test= split_data(df, param['label'])
    cv:int = 5
    # combining gridsearch of best param together with cross fold val. Hence the amount of fitting is no. of combination
    # of parameters x Kfold. As below it's params(2x2) x cv(5) = 20. Hence train model 10 times. 
    grid = GridSearchCV(define_pipeline(args), define_grid(), n_jobs=-1, cv=cv, verbose=(4 if args.verbose else 0))
    grid.fit(X_train, y_train)

   
    filename:str = f"S{prefix}{subject}E{experiment}.pkl" 
    
    # INDIVIDUAL TRAINING
    # ----------------------------------------------------------
    # training results included in grid.cv_results_
    if train_all == False:
        result = grid.cv_results_
        # parsing cross validation score across the folds. Use index of best params combination - rank 1 in key('rank_test_score') 
        # and use that to get score of each fold -> key(ie 'split0', 'split1'). 
        split:list = []
        for i in range(cv):
            split.append(f"split{i}")
        cv_score:list = [round(float(result[value + '_test_score'][result['rank_test_score'].tolist().index(1)]),4) for value in split]
        print(cv_score)
        print(f"cross_val_score: {np.array(cv_score).mean():.4f}")
        print(f"saving estimator {filename} in {MODEL_PATH}")
    # TRAIN ALL
    # -------------------------------------------------------------
    else:
        score:float = grid.score(X_test, y_test)
        print(f"experiment {experiment}: subject {subject}: accuracy = {score}")
        indiv_score.append(score)

    # save estimator
    with open(f"{MODEL_PATH}{filename}", "wb") as file:
        pickle.dump(grid, file)
    

def stream_prediction_data(X_test:pd.DataFrame, y_test:pd.DataFrame, grid:GridSearchCV)->tuple:
    # Iterate over both DataFrame and Series in batches
    accuracy_list:list = []
    for start in range(0, len(X_test), BATCH_SIZE):
        # Get a batch from X_test and y_test
        end = min(start + BATCH_SIZE, len(X_test))
        X_batch = X_test.iloc[start:end]
        y_batch = y_test.iloc[start:end]
        prediction = grid.predict(X_batch)
        accuracy_score = grid.score(X_batch, y_batch)
        time.sleep(2)
        print(f"Score on test batch {start} to {end}: {accuracy_score}")
        accuracy_list.append(accuracy_score)
        print(f"Prediction for test batch {start} to {end} = {prediction}")
    print(f"\n--------------------------------------------------")
    print(f"Mean accuracy:{np.array(accuracy_list).mean()}")

def predict(args:argparse.Namespace, streaming:bool) -> None:
    '''predict using existing pre trained model and streaming test data'''
    prefix:str = get_prefix(args.subject)
    filePATH:str = f"{MODEL_PATH}S{prefix}{args.subject}E{args.experiment}.pkl"
    
    if os.path.exists(filePATH):
        with open(filePATH,"rb") as file:
            grid = pickle.load(file)
    else:
        print("Model does not exist. Please train specific model first")
        print("exiting..")
        exit()
    param:dict = get_param(args.experiment)
    prefix:str = get_prefix(args.subject)
    # reading preprocessed file as per args params
    try:
        df:pd.DataFrame = pd.read_csv(f"{PREPROCESSED_PATH}S{prefix}{args.subject}E{args.experiment}.csv")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("exiting..")
        exit()
    _, X_test, _, y_test = split_data(df, param['label'])
    if streaming:
        stream_prediction_data(X_test, y_test, grid)
 


def main():
    '''main script for 1) train, 2) predict and 3) train all by subject and experiments'''
    args:argparse.Namespace = define_args()
    if args.mode == 'train':
        train(args.experiment, args.subject, args, False, None)
    elif args.mode == 'predict':
        predict(args, True)
    elif args.mode == 'clear':
        try:
            shutil.rmtree(MEMORY_CACHE_PATH)
            print("memory cache cleared")
        except Exception as e:
            print(f"Error occured :{e}")
            return
    else:
        overall_score:list = []
        for i in range(1):
            indiv_score:list = []
            for j in range(2):
                train(i + 1, j + 1, args, True, indiv_score)
            mean_score:float = np.array(indiv_score).mean()
            overall_score.append(mean_score)
            indiv_score = []
        print("\nMean accuracy of the six different experiments for all 109 subjects:")
        for i, score in enumerate(overall_score):
            print(f"experiment {i + 1}:        accuracy = {score}")
        print(f"Mean accuracy of {i + 1} experiments: {np.array(overall_score).mean()}")




if __name__ == '__main__':
    main()

