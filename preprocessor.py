'''script to preprocess eeg data and to visualize preprocessing pipeline'''
import mne
import matplotlib.pyplot as plt
import numpy as np
import argparse
from autoreject import AutoReject

def define_args() -> argparse.Namespace:
    """define arguments in command line"""
    parser = argparse.ArgumentParser(description="A preprocessing script to for eeg data preprocessing.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparser_indiv = subparsers.add_parser("individual", help="option for individual eeg files preprocessing split by subject and experiment")
    subparser_indiv.add_argument("--subject",default=1, type=int, help="subject no. (1 to 109). Default=1")
    subparser_indiv.add_argument("--experiment",default=3, choices=[3,4,5,6],type=int, help="experiment (3 - 6). Default=3")
    subparser_indiv.add_argument("--path",default="./data/files/", type=str, help="Default='./data/files'")
    subparser_indiv.add_argument("--visualize", default = True, action="store_true", help="enable graph visualization. Default=True")
    subparser_batch = subparsers.add_parser("batch", help="option for batch eeg files preprocessing split by subject and experiment")
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
    return True if args.visualize == True else False

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

def get_ica(epochs:mne.Epochs, visualize:bool)->mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(random_state=42, n_components=0.99)
    ar = get_ar(epochs)
    print("Starting AutoReject fit..")
    ar.fit(epochs)
    print("AutoReject fit completed")
    reject_log = ar.get_reject_log(epochs)
    print("Bad epochs from AutoReject:\n", reject_log.bad_epochs)
    if visualize:
        _, ax = plt.subplots(figsize=[15, 5])
        reject_log.plot('horizontal', ax=ax, aspect='auto')
        plt.show()
    print("Starting ICA fit....")
    ica.fit(epochs[~reject_log.bad_epochs], decim=3)
    print("ICA fit completed")
    if visualize:
        ica.plot_components()
    ica.exclude = []
    num_excl = 0
    max_ic = 2
    z_thresh = 3.5
    z_step = .05

    while num_excl < max_ic:
        eog_indices, _ = ica.find_bads_eog(epochs,
                                            ch_name=['Fp1', 'Fp2', 'F7', 'F8'],
                                            threshold=z_thresh
                                            )
        num_excl = len(eog_indices)
        z_thresh -= z_step

    ica.exclude = eog_indices
    print("Eog components excluded:\n", ica.exclude)
    return ica

def indiv_preprocessing(args:argparse.Namespace)->None:
    '''indiv processing'''
    prefix:str = ""
    if args.subject < 10:
        prefix = "00"
    elif args.subject < 100:
        prefix = "0"
    else:
        prefix = ""
    file:str = f"S{prefix}{args.subject}R{'0' if args.experiment < 10 else ''}{args.experiment}.edf" 
    raw:mne.io.Raw = mne.io.read_raw_edf(f"{args.path}S{prefix}{args.subject}/{file}",preload=True)
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
    ica = get_ica(epochs, args.visualize)
    epochs_postica = ica.apply(epochs.copy())
    if check_visual(args):
        epochs_postica.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epochs Post ICA")
    
    ar = get_ar(epochs_postica)
    print("Starting Final AutoReject fit..")
    ar.fit(epochs_postica) 
    print("Final AutoReject fit completed")
    reject_log = ar.get_reject_log(epochs_postica)
    print("Bad epochs from Final AutoReject:\n", reject_log.bad_epochs)
    if check_visual(args):
        _, ax = plt.subplots(figsize=[15, 5])
        reject_log.plot('horizontal', ax=ax, aspect='auto')
        plt.show()
    epochs_clean = ar.transform(epochs_postica)
    if check_visual(args):
        epochs_clean.plot(scalings={"eeg":100e-6}, block=True, n_epochs=7, events=True, title="Clean Epoch after Final AutoReject and ICA EOG preprocessing")
    
    


def main()-> None:
    '''main function for preprocessing'''
    mne.set_log_level(verbose="CRITICAL")
    args:argparse.Namespace = define_args()
    if (args.mode == "individual"):
        indiv_preprocessing(args)


if __name__ == '__main__':
    main()