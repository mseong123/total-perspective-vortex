
import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from mne.preprocessing import ICA
from autoreject import AutoReject
import pandas as pd


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
    try:
        os.mkdir("plot/")
    except FileExistsError:
        pass

    result = []
    for i in range(1,2):
        y = np.array([])
        X = np.array([[]])
        pca = PCA(random_state=42, n_components=32)
        prefix=""
        all_data=None
        if i < 10:
            prefix = "00"
        elif i < 100:
            prefix = "0"
        else:
            prefix = ""
        for j in range(3):
            experiment_no:int = 6 + j + (3 * j)
            raw = mne.io.read_raw_edf(f"./data/files/S{prefix}{str(i)}/S{prefix}{str(i)}R{'0' if experiment_no < 10 else ''}{experiment_no}.edf",preload=True)
            configure_channel_location(raw)
            raw_filtered:mne.io.Raw = raw.copy().filter(0.1, 30)
           

            # if i == 1 and j ==0:

            ica = ICA(random_state=42, n_components=0.99)
           
            # baseline=None means using entire epoch as baseline for correction, if use baseline=(0,0) no baseline correction

            epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=-0.1, tmax=1.0,baseline=(None,0),preload=True)
            # mask:list[bool]=[True if id == 'T0' else False for id in raw_filtered.annotations.description]
            # epochs.drop(mask)

            
            # epochs.plot(scalings={"eeg":100e-6})
            ar = AutoReject(
                n_interpolate=[1, 2, 4],
                random_state=42,
                picks=mne.pick_types(epochs.info, 
                                     eeg=True,
                                     eog=False
                                    ),
                n_jobs=-1, 
                verbose=False)
            ar.fit(epochs)
            reject_log = ar.get_reject_log(epochs)
            epochs=ar.transform(epochs)
            # fig, ax = plt.subplots(figsize=[15, 10])
            # reject_log.plot('horizontal', ax=ax, aspect='auto')
            # epochs.plot(scalings={"eeg":100e-6})
            # plt.show()
            print(reject_log.bad_epochs)
            epochs_ica = ica.fit(epochs)
            eog_indices,scores = epochs_ica.find_bads_eog(epochs, ch_name=['Fp1','Fpz','Fp2','AF7','AF8'])
            # muscle_indices,scores = epochs_ica.find_bads_muscle(epochs)
            epochs_ica.exclude += eog_indices
            # epochs_ica.exclude+=muscle_indices
            epochs = epochs_ica.apply(epochs)
            ar = AutoReject(
                n_interpolate=[1, 2, 4],
                random_state=42,
                picks=mne.pick_types(epochs.info, 
                                     eeg=True,
                                     eog=False
                                    ),
                n_jobs=-1, 
                verbose=False)
            epochs = ar.fit_transform(epochs)
            df=epochs.to_data_frame()
            if (all_data is None):
                all_data = df
            else:
                all_data = pd.concat([all_data,df])
            all_data.to_csv("hello.csv")
            y = np.hstack((y,np.array([event[2] for event in epochs.events])))
            # fig, ax = plt.subplots(figsize=[15, 10])
            # reject_log.plot('horizontal', ax=ax, aspect='auto')
            # epochs.plot(scalings={"eeg":100e-6})
            # plt.show()
            data = []
            for idx,value in enumerate(epochs.get_data()):
                # # if idx==0:
                # #     data.append(pca.fit_transform(value).tolist())
                # else:
                data.append(pca.fit_transform(value).tolist())
                # data.append(pca.fit_transform(value).tolist())
            data = np.array(data)
            if j == 0:
                # X = data.reshape(data.shape[0],-1)
                X = data
            else:
                # X = np.vstack((X,data.reshape(data.shape[0],-1)))
                X = np.vstack((X,data))
        # X = X.transpose([0,2,1])
        # reduced_data = []
        # pca2 = PCA(random_state=42, n_components=32)
        # for sample in X:
        #     reduced_sample = pca2.fit_transform(sample)
        #     reduced_data.append(reduced_sample)
        # reduced_data = np.array(reduced_data)
        # X = reduced_data.transpose(0,2,1)
        X = X.reshape(X.shape[0], -1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        # cv_scores=cross_val_score(classifier, X_train, y_train, cv=5)
        # print("Cross-validation scores:", cv_scores)
        # print("Mean cross-validation score:", cv_scores.mean())

        classifier = MLPClassifier(random_state=42, alpha=0.001, verbose=1)
        classifier.fit(X_train, y_train)
        predict = classifier.predict(X_test)
        print("response")
        print([predict == y_test])
        t1=predict[predict==1]
        t2=predict[predict==2]
        t3=predict[predict==3]
        print(f"participant {i}")
        print(t1)
        print(t2)
        print(t3)
        score=classifier.score(X_test,y_test)
        print(score)
        result.append(score)



        # CV = StratifiedKFold(n_splits=5)
        # LR = LogisticRegressionCV(random_state=42, solver="saga", cv=CV, refit=True)
        # LR.fit(X_train,y_train)
        # print(LR.scores_)
        # predict = LR.predict(X_test)
        # t1=predict[predict==1]
        # t2=predict[predict==2]
        # t3=predict[predict==3]
        # print(f"participant {i}")
        # print(t1)
        # print(t2)
        # print(t3)
        # score = LR.score(X_test, y_test)
        # print(f"score: {score}")
        # result.append(score)
        # print(LR.predict(X_test))


    print(result)
    print(np.array(result).mean())


    # raw = mne.io.read_raw_edf(f"./data/files/S004/S004R04.edf",preload=True)
    # configure_channel_location(raw)
    # raw_filtered = raw.copy().filter(8, 40)
    # print(raw.get_data().shape)
    # epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=0, tmax=np.round(raw_filtered.annotations.duration.mean(),2) - 0.01,baseline=(None,None),preload=True)
    # X = epochs.get_data().reshape(epochs.get_data().shape[0],-1)
    # y = raw_filtered.annotations.description
    # All below random_state is for different reasons
    # # print("Cs",LR.Cs_)
    # print("Coefficients",LR.coef_.shape)
    # print("Clases",LR.classes_)

    # plotting
    # fig = raw.plot(block=True, clipping=None, scalings={"eeg":70e-6}, show=False)
    # fig.savefig(f"plot/raw.png", format="png")
    # fig_filtered = raw_filtered.plot(block=True, clipping=None, scalings={"eeg":70e-6}, show=False)
    # fig_filtered.savefig(f"plot/raw_filtered.png", format="png")
    # fig_raw_fft = raw.compute_psd().plot(show=False)
    # fig_raw_fft.savefig(f"plot/raw_fft.png", format="png")
    # fig_raw_filtered_fft = raw_filtered.compute_psd().plot(show=False)
    # fig_raw_filtered_fft.savefig(f"plot/raw_filtered_fft.png", format="png")

    # pca = PCA(random_state=42)
    # pca.fit(flattened_epoch_sample)
     

    
  

    


main()

