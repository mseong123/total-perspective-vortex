import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sklearn
from sklearn.neural_network import MLPClassifier
from mne.preprocessing import ICA
import pandas as pd
from sklearn.metrics import accuracy_score



from sklearn.model_selection import GridSearchCV

def configure_channel_location(raw:mne.io.Raw) -> None:
    new_channel:list[str]=['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', \
                 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', \
                'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', \
                'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', \
                'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', \
                'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', \
                'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']
    raw.rename_channels({ch_name:new_channel[i] for i,ch_name in enumerate(raw.ch_names)})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

def main() -> None:
    mne.set_log_level(verbose="CRITICAL")
    param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization type
    'solver': ['saga'],  # Supports l1, l2, and elasticnet
    'l1_ratio': [0.1, 0.5, 0.9],  # ElasticNet mix ratio
    'class_weight': ['balanced', None],  # Handle class imbalance
    }
    result = []
    for i in range(100,101):
        frame = pd.read_csv(f"./preprocessed_data/S{i}E1.csv")
        # frame = pd.read_csv(f"./data.csv")
        X = frame[(frame['condition'] == 'T1') | (frame['condition'] == "T2")]
        # target = pd.Series(frame['condition'].values)
        target = pd.Series(X['condition'].values)
        X = X.drop(["condition", "Unnamed: 0", "epoch"], axis=1)
        # X = frame.drop(["condition", "Unnamed: 0", "epoch"], axis=1)
        scaler = StandardScaler()
        pca = PCA(random_state=42, n_components=65)
        train_data, test_data, train_target, test_target = train_test_split(X, target, test_size=0.3, random_state=42, stratify=target)
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.fit_transform(test_data)
        train_data = pca.fit_transform(train_data)
        test_data = pca.fit_transform(test_data)
    # print(pd.DataFrame(train_data))
    # parameter_grid = {

	# 	"alpha": (0.1, 0.2, 0.3, 0.4, 0.5),
	# 	"momentum": (0.1, 0.2, 0.3, 0.4, 0.5)
	# }
    # classifier = RandomizedSearchCV(
	# 	estimator=MLPClassifier(max_iter=1000, random_state=69, verbose=1),
	# 	param_distributions=parameter_grid,
	# 	random_state=42,
	# 	n_jobs=4,
	# 	n_iter=5,
	# 	verbose=4,
	# )
        classifier = MLPClassifier(max_iter = 1000, random_state=42, alpha=0.2)
        classifier.fit(train_data, train_target)
        score = classifier.score(test_data,test_target)
        print("score:",score)
        result.append(score)

        # grid_search = RandomizedSearchCV(
        #     estimator=LogisticRegression(max_iter=1000),
        #     param_distributions=param_grid,
        #     cv=5,
        # )
        # grid_search.fit(train_data, train_target)

        # print("Best Parameters:", grid_search.best_params_)
        # print("score", grid_search.score(test_data, test_target))

        # CV = StratifiedKFold(n_splits=5)
        # LR = LogisticRegressionCV(random_state=42, cv=CV,Cs=[0.1,1,10], refit=True,max_iter=2000)
        # LR.fit(train_data,train_target)
        # print(LR.scores_)
        # predict = LR.predict(test_data)
        # score = LR.score(test_data, test_target)
        # print(f"score: {score}")
        # print(f"accuracy_score{accuracy_score(test_target,predict)}")
        # result.append(score)

    print(result)
    print(np.array(result).mean())



main()