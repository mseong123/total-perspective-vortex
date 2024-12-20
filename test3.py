import mne
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import sklearn
from sklearn.neural_network import MLPClassifier
from mne.preprocessing import ICA
from autoreject import AutoReject





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
    raw = mne.io.read_raw_edf("./data/files/S001/S001R03.edf",preload=True)

    configure_channel_location(raw)
    # fig = raw.plot(scalings={'eeg':100e-6},block=True, n_channels=30)
    ica = ICA(random_state=42, n_components=0.99)
    # epochs = mne.Epochs(raw, tmin=-0.2, tmax=raw.annotations.duration.mean(), baseline=(None,0), preload=True)
    epochs.plot(scalings={"eeg":100e-6},block=True, events=True)
    # fig = raw.compute_psd().plot()
    # fig.savefig('./plot/raw_psd')
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
    reject = ar.get_reject_log(epochs)
    print(reject.bad_epochs)
    ica.fit(epochs[~reject.bad_epochs], decim=3)
    
    

   

main()