
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
    for i in range(1,99):
        prefix=""
        all_data=None
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
            raw_filtered:mne.io.Raw = raw.copy().filter(0.1, 30)
           


            ica = ICA(random_state=42, n_components=0.99)
            res = raw.copy().filter(1,30)
            events_ica = mne.make_fixed_length_events(res, duration=1)
            epochs_ica = mne.Epochs(res,events_ica, tmin=0, tmax=1.0,baseline=None,preload=True)
            
            ar = AutoReject(
                n_interpolate=[1, 2, 4],
                random_state=42,
                picks=mne.pick_types(epochs_ica.info, 
                                     eeg=True,
                                     eog=False
                                    ),
                n_jobs=-1, 
                verbose=False)
            ar.fit(epochs_ica)
            reject_log = ar.get_reject_log(epochs_ica)
            ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)
            ica.exclude = []
            num_excl = 0
            max_ic = 2
            z_thresh = 3.5
            z_step = .05

            while num_excl < max_ic:
                eog_indices, eog_scores = ica.find_bads_eog(epochs_ica,
													ch_name=['Fp1', 'Fp2', 'F7', 'F8'], 
													threshold=z_thresh
													)
                num_excl = len(eog_indices)
                z_thresh -= z_step

            ica.exclude = eog_indices
            print(ica.exclude)
            epochs = mne.Epochs(raw_filtered, tmin=-0.1, tmax=1.0,baseline=(None,0),preload=True)
            epochs_postica = ica.apply(epochs.copy())
            ar = AutoReject(
                n_interpolate=[1, 2, 4],
                random_state=42,
                picks=mne.pick_types(epochs_postica.info, 
                                     eeg=True,
                                     eog=False
                                    ),
                n_jobs=-1, 
                verbose=False)
            epochs_clean = ar.fit_transform(epochs_postica)
            df=epochs_clean.to_data_frame()
            if (all_data is None):
                all_data = df
            else:
                all_data = pd.concat([all_data,df])
            all_data.to_csv(f"morning{i}.csv")     

    
  

    


main()

