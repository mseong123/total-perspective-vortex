import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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
    for i in range(1,110):
        y = np.array([])
        X = np.array([[]])
        prefix=""
        if i < 10:
            prefix = "00"
        elif i < 100:
            prefix = "0"
        else:
            prefix = ""
        for j in range(3):
            experiment_no:int = 3 + j + (3 * j)
            raw = mne.io.read_raw_edf(f"./data/files/S{prefix}{str(i)}/S{prefix}{str(i)}R{'0' if experiment_no < 10 else ''}{experiment_no}.edf",preload=True)
            configure_channel_location(raw)
            raw_filtered:mne.io.Raw = raw.copy().filter(8, 40)
            # baseline=None means using entire epoch as baseline for correction, if use baseline=(0,0) no baseline correction
            epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=0, tmax=raw_filtered.annotations.duration.mean(),baseline=(0,0),preload=True)
            y = np.hstack((y, [str(id) for id in raw_filtered.annotations.description if id == 'T1' or id == 'T2']))
            if (len(epochs.get_data())==29):
                y=y[:-1]

            mask:list[bool]=[True if id == 'T0' else False for id in raw_filtered.annotations.description]
            epochs.drop(mask)
           
            if j == 0:
                X = epochs.get_data().reshape(epochs.get_data().shape[0],-1)
            else:
                X = np.vstack((X,epochs.get_data().reshape(epochs.get_data().shape[0],-1)))

        pca = PCA(random_state=42)
        X=pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        # Don't need put random_state for StratifiedKFold because no shuffling takes place before splitting, samples split as it is.
        CV = StratifiedKFold(n_splits=10)
        LR = LogisticRegressionCV(random_state=42, cv=CV, refit=False)
        LR.fit(X_train,y_train)
        # print(LR.scores_)
        print(LR.score(X_test, y_test)) 
        result.append(LR.score(X_test, y_test))
        # result.append(LR.predict(X_test))
        # print("iter",LR.n_iter_)

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

