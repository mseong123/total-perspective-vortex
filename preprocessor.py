'''script to preprocess eeg data and to visualize preprocessing pipeline'''

import argparse
import os
import mne
import matplotlib.pyplot as plt
import pandas as pd
from autoreject import AutoReject
from param import RANDOM_STATE, get_param, PREPROCESSED_PATH, DATA_PATH

def define_args() -> argparse.Namespace:
    """define arguments in command line"""
    parser = argparse.ArgumentParser(description="A preprocessing script for eeg data preprocessing.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--start",default=1, type=int, help=(
        "start of subject no. (1 to 109)\n"
        "Default=1")
    )
    parser.add_argument("--end",default=109, type=int, help=(
        "end of subject no. (1 to 109)\n"
        "Default=109")
    )
    parser.add_argument("--experiment",default=1, choices=[1,2,3,4,5,6], type=int, help=(
            "experiment 1: left first vs right fist\n"
            "experiment 2: imagine left fist vs right fist\n"
            "experiment 3: both fists vs both feet\n"
            "experiment 4: imagine both fists vs both feet\n"
            "experiment 5: rest vs left fist\n"
            "experiment 6: rest vs imagine both feet\n"
            "Default=1")
            )
    parser.add_argument("--visualize", action='store_true', help="enable eeg data visualization.")
    return parser.parse_args()

def configure_channel_location(raw:mne.io.Raw) -> None:
    '''setting channel location for raw data'''
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

def get_ar(epochs:mne.Epochs) -> AutoReject:
    '''create Autoreject object'''
    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        random_state=RANDOM_STATE,
        picks=mne.pick_types(epochs.info,
                                eeg=True,
                                eog=False
        ),
        n_jobs=-1,
        verbose=False)
    return ar

def get_ica(epochs:mne.Epochs, args:argparse.Namespace)->mne.preprocessing.ICA:
    '''fit ICA, find_bads_eog and return ICA'''
    ica:mne.preprocessing.ICA = mne.preprocessing.ICA(random_state=RANDOM_STATE, n_components=0.99)
    ar:AutoReject = get_ar(epochs)
    print("Starting AutoReject fit..")
    ar.fit(epochs)
    print("AutoReject fit completed")

    reject_log = ar.get_reject_log(epochs)
    print("Bad epochs from AutoReject:\n", reject_log.bad_epochs)

    if args.visualize:
        _, ax = plt.subplots(figsize=[15, 5])
        reject_log.plot('horizontal', ax=ax, aspect='auto')
        plt.show()

    print("Starting ICA fit....")
    ica.fit(epochs[~reject_log.bad_epochs], decim=3)
    print("ICA fit completed")
    
    if args.visualize:
        ica.plot_components()
    ica.exclude = []
    num_excl:int = 0
    max_ic:int = 2
    z_thresh:float = 3.5
    z_step:float = .05

    while num_excl < max_ic:
        eog_indices, _ = ica.find_bads_eog(epochs,
                                            ch_name=['Fp1', 'Fp2', 'F7', 'F8'],
                                            threshold=z_thresh
                                            )
        num_excl:int = len(eog_indices)
        z_thresh -= z_step

    ica.exclude = eog_indices
    print("EOG components excluded:\n", ica.exclude)
    return ica
    
def preprocessing(args:argparse.Namespace, param:dict) -> None:
    '''Preprocessing of eeg data and save result as pandas DataFrame'''
    data:pd.DataFrame | None = None
    for i in range(args.start, args.end + 1):
        print(f"Preprocessing subject {i}, experiment: {args.experiment}")
        prefix:str = ""
        if i < 10:
            prefix = "00"
        elif i < 100:
            prefix = "0"
        else:
            prefix = ""
        for j, file_no in enumerate(param["file_no"]):
            print(f"Experiment Run {j + 1}")
            file:str = f"S{prefix}{i}R{'0' if file_no < 10 else ''}{file_no}"
            raw:mne.io.Raw = mne.io.read_raw_edf(f"{DATA_PATH}S{prefix}{i}/{file}.edf",preload=True)
            configure_channel_location(raw)

            if args.visualize:
                raw.plot(scalings={"eeg":100e-6}, title=f"Time Series Plot {file} before low and high pass filter")
                raw.compute_psd().plot()

            raw_filtered:mne.io.Raw = raw.copy().filter(1,40)

            if args.visualize:
                raw_filtered.compute_psd().plot()
                raw_filtered.plot(scalings={"eeg":100e-6}, title=f"Time Series Plot {file} after low and high pass filter", block=True)

            epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=0, tmax=raw_filtered.annotations.duration.mean(),baseline=(0,0), preload=True)

            if args.visualize:
                epochs.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epoch Time Series Plot before preprocessing")

            ica:mne.preprocessing.ICA = get_ica(epochs, args)
            epochs_postica:mne.Epochs = ica.apply(epochs.copy())

            if args.visualize:
                epochs_postica.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epoch Time Series Plot Post ICA")

            ar:AutoReject = get_ar(epochs_postica)

            print("Starting Final AutoReject fit..")
            ar.fit(epochs_postica.copy())
            print("Final AutoReject fit completed")

            reject_log = ar.get_reject_log(epochs_postica)
            print("Bad epochs from Final AutoReject:\n", reject_log.bad_epochs)

            if args.visualize:
                _, ax = plt.subplots(figsize=[15, 5])
                reject_log.plot('horizontal', ax=ax, aspect='auto')
                plt.show()

            epochs_clean:mne.Epochs = ar.transform(epochs_postica.copy())

            if args.visualize:
                epochs_clean.plot(scalings={"eeg":100e-6}, block=True, n_epochs=7, events=True, title="Clean Epoch after Final AutoReject and ICA EOG preprocessing")

            df:pd.DataFrame = epochs_clean.to_data_frame()
            if data is None:
                data = df
            else:
                data = pd.concat([data,df])
        print(f"Saving file S{prefix}{i}E{args.experiment}.csv in {PREPROCESSED_PATH}")
        data.to_csv(f"{PREPROCESSED_PATH}S{prefix}{i}E{args.experiment}.csv")



def main()-> None:
    '''main function for preprocessing'''
    mne.set_log_level(verbose="CRITICAL")
    try:
        os.mkdir(PREPROCESSED_PATH)
    except FileExistsError:
        pass
    args:argparse.Namespace = define_args()
    param = get_param(args.experiment)
    preprocessing(args, param)


if __name__ == '__main__':
    main()
