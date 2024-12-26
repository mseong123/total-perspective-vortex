'''script to preprocess eeg data and to visualize preprocessing pipeline'''

import argparse
import os
import mne
import matplotlib.pyplot as plt
import pandas as pd
from autoreject import AutoReject

def define_args() -> argparse.Namespace:
    """define arguments in command line"""
    parser = argparse.ArgumentParser(description="A preprocessing script for eeg data preprocessing.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparser_indiv = subparsers.add_parser("individual", help="option for individual eeg files preprocessing split by subject and experiment")
    subparser_indiv.add_argument("--subject",default=1, type=int, help="subject no. (1 to 109). Default=1")
    subparser_indiv.add_argument("--experiment",default=3, choices=[3,4,5,6],type=int, help="experiment (3 - 6). Default=3")
    subparser_indiv.add_argument("--path",default="./data/files/", type=str, help="Default='./data/files'")
    subparser_indiv.add_argument("--visualize", action='store_true', help="enable graph visualization.")
    subparser_batch = subparsers.add_parser("batch", help="option for batch eeg files preprocessing split by subject and experiment. Processed file includes all 3 runs of experiment.")
    subparser_batch.add_argument("--start",default=1, type=int, help="start of subject no. (1 to 109). Default=1")
    subparser_batch.add_argument("--end",default=109, type=int, help="end of subject no. (1 to 109). Default=109")
    subparser_batch.add_argument("--experiment",default=3, choices=[3,4,5,6], type=int, help="experiment (3 - 6). Default=3")
    subparser_batch.add_argument("--path",default="./data/files/", type=str, help="Default='./data/files'")
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

def check_visual(args)->bool:
    '''to check if visualization argument is true'''
    return True if args.visualize is True else False

def get_ar(epochs:mne.Epochs) -> AutoReject:
    '''create Autoreject object'''
    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        random_state=42,
        picks=mne.pick_types(epochs.info,
                                eeg=True,
                                eog=False
        ),
        n_jobs=-1,
        verbose=False)
    return ar

def get_ica(epochs:mne.Epochs, args:argparse.Namespace)->mne.preprocessing.ICA:
    '''fit, find_bads_eog and return ICA'''
    ica:mne.preprocessing.ICA = mne.preprocessing.ICA(random_state=42, n_components=0.99)
    ar:AutoReject = get_ar(epochs)
    if args.mode == "individual":
        print("Starting AutoReject fit..")

    ar.fit(epochs)

    if args.mode == "individual":
        print("AutoReject fit completed")

    reject_log = ar.get_reject_log(epochs)

    if args.mode == "individual":
        print("Bad epochs from AutoReject:\n", reject_log.bad_epochs)
    if args.mode == "individual" and args.visualize:
        _, ax = plt.subplots(figsize=[15, 5])
        reject_log.plot('horizontal', ax=ax, aspect='auto')
        plt.show()

    if args.mode == "individual":
        print("Starting ICA fit....")

    ica.fit(epochs[~reject_log.bad_epochs], decim=3)

    if args.mode == "individual":
        print("ICA fit completed")
    if args.mode == "individual" and args.visualize:
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
    if args.mode == "individual":
        print("EOG components excluded:\n", ica.exclude)
    return ica

def indiv_preprocessing(args:argparse.Namespace, path:str)->None:
    '''indiv processing'''
    prefix:str = ""
    if args.subject < 10:
        prefix = "00"
    elif args.subject < 100:
        prefix = "0"
    else:
        prefix = ""

    file:str = f"S{prefix}{args.subject}R{'0' if args.experiment < 10 else ''}{args.experiment}"
    raw:mne.io.Raw = mne.io.read_raw_edf(f"{args.path}S{prefix}{args.subject}/{file}.edf",preload=True)
    configure_channel_location(raw)

    if check_visual(args):
        raw.plot(scalings={"eeg":100e-6}, title=f"Time Series Plot {file} before low and high pass filter")
        raw.compute_psd().plot()

    raw_filtered:mne.io.Raw = raw.copy().filter(1,40)

    if check_visual(args):
        raw_filtered.compute_psd().plot()
        raw_filtered.plot(scalings={"eeg":100e-6}, title=f"Time Series Plot {file} after low and high pass filter", block=True)

    epochs = mne.Epochs(raw_filtered, tmin=0, tmax=raw_filtered.annotations.duration.mean(),baseline=(0,0), preload=True)

    if check_visual(args):
        epochs.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epoch before preprocessing")

    ica:mne.preprocessing.ICA = get_ica(epochs, args)
    epochs_postica:mne.Epochs = ica.apply(epochs.copy())

    if check_visual(args):
        epochs_postica.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epochs Post ICA")
    ar:AutoReject = get_ar(epochs_postica)

    print("Starting Final AutoReject fit..")
    ar.fit(epochs_postica.copy())
    print("Final AutoReject fit completed")

    reject_log = ar.get_reject_log(epochs_postica)
    print("Bad epochs from Final AutoReject:\n", reject_log.bad_epochs)
    if check_visual(args):
        _, ax = plt.subplots(figsize=[15, 5])
        reject_log.plot('horizontal', ax=ax, aspect='auto')
        plt.show()

    epochs_clean:mne.Epochs = ar.transform(epochs_postica.copy())
    if check_visual(args):
        epochs_clean.plot(scalings={"eeg":100e-6}, block=True, n_epochs=7, events=True, title="Clean Epoch after Final AutoReject and ICA EOG preprocessing")

    print(f"Saving file {file}_indiv.csv in path: {path}")
    epochs_clean.to_data_frame().to_csv(f"{path}{file}_indiv.csv")

def batch_preprocessing(args:argparse.Namespace, path:str) -> None:
    '''batch processing'''
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
        for j in range(3):
            print(f"Experiment Run {j + 1}")
            experiment_no:int = args.experiment + j + (3 * j)
            file:str = f"S{prefix}{i}R{'0' if experiment_no < 10 else ''}{experiment_no}"
            raw:mne.io.Raw = mne.io.read_raw_edf(f"{args.path}S{prefix}{i}/{file}.edf",preload=True)
            configure_channel_location(raw)

            raw_filtered:mne.io.Raw = raw.copy().filter(1,40)
            epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=0, tmax=raw_filtered.annotations.duration.mean(),baseline=(0,0), preload=True)

            ica:mne.preprocessing.ICA = get_ica(epochs, args)
            epochs_postica:mne.Epochs = ica.apply(epochs.copy())
            ar:AutoReject = get_ar(epochs_postica)
            epochs_clean = ar.fit_transform(epochs_postica.copy())

            df:pd.DataFrame = epochs_clean.to_data_frame()
            if data is None:
                data = df
            else:
                data = pd.concat([data,df])
        print(f"Saving file S{prefix}{i}R{'0' if args.experiment < 10 else ''}{args.experiment}.csv in path: {path}")
        data.to_csv(f"{path}S{prefix}{i}R{'0' if args.experiment < 10 else ''}{args.experiment}.csv")



def main()-> None:
    '''main function for preprocessing'''
    mne.set_log_level(verbose="CRITICAL")
    preproc_indiv_path = "preprocessed_data_indiv/"
    preproc_batch_path = "preprocessed_data_batch/"
    try:
        os.mkdir(preproc_indiv_path)
        os.mkdir(preproc_batch_path)
    except FileExistsError:
        pass
    args:argparse.Namespace = define_args()
    if args.mode == "individual":
        indiv_preprocessing(args, preproc_indiv_path)
    else:
        batch_preprocessing(args, preproc_batch_path)


if __name__ == '__main__':
    main()
