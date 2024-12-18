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
    raw = mne.io.read_raw_edf("./data/files/S001/S001R07.edf",preload=True)
    configure_channel_location(raw)
    # raw.plot(clipping=None, scalings={"eeg":70e-6})
    raw_filtered = raw.copy().filter(8, 40)
    # variances = np.var(raw_filtered.get_data(), axis=1)
    # low_percentile = np.percentile(variances, 10)  # 10th percentile for low variance
    # high_percentile = np.percentile(variances, 90)  #
    # bad_channels = [ch for ch, var in zip(raw_filtered.ch_names, variances) if var < low_percentile or var > high_percentile]
    # print(bad_channels) 
    # raw_filtered = raw_filtered.drop_channels(bad_channels)
    epochs = mne.Epochs(raw_filtered, tmin=0, tmax=np.round(raw_filtered.annotations.duration.mean(),2),baseline=(0,0),preload=True)
    mask:list[bool]=[True if id == 'T0' or id=='T2' else False for id in raw_filtered.annotations.description]
    epochs.drop(mask) 
    epochs.plot(scalings={"eeg":100e-6})
    ica = ICA(random_state=42)
    ica.fit(raw_filtered)
    eog_indices,scores = ica.find_bads_eog(raw_filtered, ch_name=['Fp1','Fpz','Fp2','AF7','AF8'])
    muscle_indices,scores = ica.find_bads_muscle(raw_filtered)
    ica.exclude += eog_indices
    ica.exclude+=muscle_indices
    raw_filtered = ica.apply(raw_filtered)
    epochs1 = mne.Epochs(raw_filtered, tmin=0, tmax=np.round(raw_filtered.annotations.duration.mean(),2),baseline=(0,0),preload=True)
    # print([str(id) for id in raw_filtered.annotations.description if id == 'T1' or id == 'T2'])
    epochs1.drop(mask)
    epochs1.plot(scalings={"eeg":100e-6})
    
    plt.show()

main()