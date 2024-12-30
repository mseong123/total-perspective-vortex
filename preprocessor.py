'''script to preprocess eeg data and to visualize preprocessing pipeline'''

import argparse
import os
import time
import mne
import matplotlib.pyplot as plt
import pandas as pd
from autoreject import AutoReject
from param import RANDOM_STATE, get_param, get_prefix,PREPROCESSED_PATH, DATA_PATH

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
    '''setting channel location for raw data so that raw.compute_psd().plot() has channel info '''
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
    # Autoreject is a program that takes a sample of channels and epochs and check for artifacts/noises and
    # interpolate or reject them.
    ar:AutoReject = get_ar(epochs)
    print("Starting AutoReject fit..")
    # Why only fit and not transform? To preserve epochs/channel data for ICA EOG(eye movement artifacts) analysis. Identify bad epochs/channels
    # and exclude them when fitting ICA. The do a final Autoreject below to actually remove those not excluded in ICA EOG analysis.
    ar.fit(epochs)
    print("AutoReject fit completed")

    reject_log = ar.get_reject_log(epochs)
    print("Bad epochs from AutoReject:\n", reject_log.bad_epochs)

    if args.visualize:
        _, ax = plt.subplots(figsize=[15, 5])
        reject_log.plot('horizontal', ax=ax, aspect='auto')
        plt.show()

    print("Starting ICA fit....")
    # Apply ICA to epochs instead of Raw to better capture the independent components of each Epoch. Otherwise
    # the entire time series of raw will skew the components.
    # The fit method apply a unmixing matrix.
    ica.fit(epochs[~reject_log.bad_epochs], decim=3)
    print("ICA fit completed")
    
    if args.visualize:
        ica.plot_components()
    ica.exclude = []
    num_excl:int = 0
    max_ic:int = 2
    z_thresh:float = 3.5
    z_step:float = .05
    # lowering threshold to detect EOG artifacts (at least 2 ICA components). Since my filter low pass cutoff is at 1Hz,
    # will definitely pick up relevant artifacts. 
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
    for i in range(args.start, args.end + 1):
        data:pd.DataFrame | None = None
        print(f"Preprocessing subject {i}, experiment: {args.experiment}")
        start_time = time.time()
        prefix:str = get_prefix(i) 
        for j, file_no in enumerate(param["file_no"]):
            # 3 runs per experiment
            print(f"\nExperiment Run {j + 1}")
            file:str = f"S{prefix}{i}R{'0' if file_no < 10 else ''}{file_no}"
            raw:mne.io.Raw = mne.io.read_raw_edf(f"{DATA_PATH}S{prefix}{i}/{file}.edf",preload=True)
            configure_channel_location(raw)

            if args.visualize:
                raw.plot(scalings={"eeg":100e-6}, title=f"Time Series Plot {file} before low and high pass filter")
                raw.compute_psd().plot()

            # first part of preprocessing. Filter out slow drifts (low frequency with high amplitude 
            # relative to brain voltage (in micro volts). Example is background electronic interference.
            # Also filter out high frequency > 40 hertz. Neuron activities related to motor in theory should be alpha
            # and beta waves. If i filter using 8 hz to 40 hz (as per eval sheet), i get better result but i filter out at lower rate -1hz to show
            # effect of EOG ICA filtering below and Autoreject (repairing channels/epochs).

            # Also filter at raw level and not at epoch level since low frequency drifts might happen across multiple
            # epochs. More accurate to filter out at raw then only separate them into Epochs. 
            # The filtering algorithm uses sampling.  
            raw_filtered:mne.io.Raw = raw.copy().filter(1,40)

            if args.visualize:
                raw_filtered.compute_psd().plot()
                # compared to raw.plot(), should look cleaner. However large amplitudes not related to brain waves
                # are still apparent in plot.
                raw_filtered.plot(scalings={"eeg":100e-6}, title=f"Time Series Plot {file} after low and high pass filter", block=True)

            # No baseline transformation, didn't take pre-Event Related Potential splits. Not relevant to classification model
            # as any ERP onset related artifacts for specific channels will be reflected across the samples and will be 
            # picked up by the models. Also take duration of entire ERP event time series instead of a sample of it (ie 1 second vs 4.15 seconds)
            epochs:mne.Epochs = mne.Epochs(raw_filtered, tmin=0, tmax=raw_filtered.annotations.duration.mean(),baseline=(0,0), preload=True)
            
            if args.visualize:
                # Plot can be adjusted to show number of epochs, but in general should look the same as raw_filtered.plot() 
                # above although more concise
                epochs.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epoch Time Series Plot before preprocessing")

            ica:mne.preprocessing.ICA = get_ica(epochs, args)
            # .apply method basically applying the unmixing matrix learnt above to epoch data then 
            # 'switching off' the time series of the detected EOG artifacts component - setting them to zero, in ICA matrix form
            # then reapply mixing matrix and transforming back to Epochs data.
            epochs_postica:mne.Epochs = ica.apply(epochs.copy())

            if args.visualize:
                # Will show cleaner plot vs epochs above but will still show some epochs with artifacts not removed
                epochs_postica.plot(scalings={"eeg":100e-6}, n_epochs=7, events=True, title="Epoch Time Series Plot Post ICA")

            ar:AutoReject = get_ar(epochs_postica)

            print("Starting Final AutoReject fit..")
            # final Autoreject to remove channel/epochs not removed above and also to interpolate bad data (smoothen them out
            # with neighbouring data)
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
                # Final epoch data
                epochs_clean.plot(scalings={"eeg":100e-6}, block=True, n_epochs=7, events=True, title="Clean Epoch after Final AutoReject and ICA EOG preprocessing")

            df:pd.DataFrame = epochs_clean.to_data_frame()
            if data is None:
                data = df
            else:
                data = pd.concat([data,df])
        print(f"Saving file S{prefix}{i}E{args.experiment}.csv in {PREPROCESSED_PATH}\n")
        print(f"Time Elapsed:{(time.time() - start_time):.0f} seconds\n")

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
