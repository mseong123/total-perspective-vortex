import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from sklearn.decomposition import PCA
from mne.preprocessing import ICA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sklearn
from sklearn.neural_network import MLPClassifier

sklearn.set_config(enable_metadata_routing=True)

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
    for i in range(1,30):
        y = np.array([])
        X = np.array([[]])
        pca = PCA(random_state=42)
        groups = []
        prefix=""
        X_train_1 = []
        X_train_2 = []
        X_test = []
        y_train_1 = []
        y_train_2 = []
        y_test = []
        if i < 10:
            prefix = "00"
        elif i < 100:
            prefix = "0"
        else:
            prefix = ""
        for j in range(3):
            experiment_no:int = 4 + j + (3 * j)
            raw = mne.io.read_raw_edf(f"./data/files/S{prefix}{str(i)}/S{prefix}{str(i)}R{'0' if experiment_no < 10 else ''}{experiment_no}.edf",preload=True)
            configure_channel_location(raw)
            raw_filtered:mne.io.Raw = raw.copy().filter(8, 40)
            print('shape',raw_filtered.get_data().shape)

            # ica = ICA(random_state=42)
            # ica.fit(raw_filtered)
            # eog_indices,scores = ica.find_bads_eog(raw_filtered, ch_name=['Fp1','Fpz','Fp2','AF7','AF8'])
            # muscle_indices,scores = ica.find_bads_muscle(raw_filtered)
            # ica.exclude += eog_indices
            # ica.exclude+=muscle_indices
            # raw_filtered = ica.apply(raw_filtered)
            # variances = np.var(raw_filtered.get_data(), axis=1)
            # low_percentile = np.percentile(variances, 5)  # 10th percentile for low variance
            # high_percentile = np.percentile(variances, 95)  #
            # bad_channels = [ch for ch, var in zip(raw_filtered.ch_names, variances) if var < low_percentile or var > high_percentile]
            # raw_filtered = raw_filtered.drop_channels(bad_channels)
            # print(len(raw_filtered.ch_names))
            # baseline=None means using entire epoch as baseline for correction, if use baseline=(0,0) no baseline correction
            epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=0, tmax=raw_filtered.annotations.duration.mean(),baseline=None,preload=True)
            # y = np.hstack((y, [str(id) for id in raw_filtered.annotations.description if id == 'T1' or id == 'T2']))
            y = [str(id) for id in raw_filtered.annotations.description if id == 'T1' or id == 'T2']

            if (len(epochs.get_data())==29):
                y=y[:-1]

            mask:list[bool]=[True if id == 'T0' else False for id in raw_filtered.annotations.description]
            epochs.drop(mask)
            # epochs.plot()
            # plt.show()
            data = []
            for idx,value in enumerate(epochs.get_data()):
                if (idx == 0):
                    data.append(pca.fit_transform(value).tolist())
                else:
                    data.append(pca.transform(value).tolist())
            data = np.array(data)
            
            # if j == 0:
            #     X = data.reshape(data.shape[0], -1)
            # else:
            #     X = np.vstack((X,data.reshape(data.shape[0],-1)))
            
            X = data.reshape(data.shape[0], -1)

            if j == 2:
                X = data.reshape(data.shape[0], -1)
                X_train_1 = X
                y_train_1 = np.array(y)
                groups.append([j] *len(X))
            elif j == 1:
                X_train_2 = X
                y_train_2=np.array(y)
                groups.append([j] *len(X))
            elif j ==0:
                X_test = X
                y_test = np.array(y)
        # X = X.transpose([0,2,1])
        # reduced_data = []
        # pca2 = PCA(random_state=42, n_components=30)
        # for sample in X:
        #     reduced_sample = pca2.fit_transform(sample)
        #     reduced_data.append(reduced_sample)
        # reduced_data = np.array(reduced_data)
        # X = reduced_data.transpose(0,2,1)
        # X = X.reshape(X.shape[0], -1)
        # print(X.shape)
        groups = [item for sublist in groups for item in sublist]
        group_kfold = GroupKFold(n_splits=2) 
        scaler = StandardScaler()
        X_train_1 = scaler.fit_transform(X_train_1)
        X_train_2 = scaler.fit_transform(X_train_2)
        X_train = np.concatenate((X_train_1, X_train_2))
        y_train = np.concatenate((y_train_1, y_train_2))
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, groups=groups)
        X_test = scaler.fit_transform(X_test)
        # Don't need put random_state for StratifiedKFold because no shuffling takes place before splitting, samples split as it is.
        CV = StratifiedKFold(n_splits=5)
        LR = LogisticRegressionCV(random_state=42, cv=group_kfold,verbose=0, solver="saga", refit=True)
        LR.fit(X_train,y_train, groups=groups)
        print(LR.scores_)
        predict = LR.predict(X_test)
        t1=predict[predict=='T1']
        t2=predict[predict=='T2']
        print(f"participant {i}")
        print(t1)
        print(t2)
        score = LR.score(X_test, y_test)
        print(f"score: {score}")
        result.append(score)
        print(LR.predict(X_test))
        # print("iter",LR.n_iter_)

    
        # classifier = MLPClassifier(random_state=42,hidden_layer_sizes=(2000,1000, 500,250), verbose=1,alpha=0.001)
        # classifier.fit(X_train, y_train)
        # predict = classifier.predict(X_test)
        # print("response")
        # print([predict == y_test]) 
        # score=classifier.score(X_test,y_test)
        # print(score)
        # t1=predict[predict=='T1']
        # t2=predict[predict=='T2']
        # print(f"participant {i}")
        # print(t1)
        # print(t2)
        # result.append(score)
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

