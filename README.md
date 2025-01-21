# total-perspective-vortex | 42KL


![Neural Network](https://github.com/mseong123/total-perspective-vortex/blob/main/images/EEG_system.jpg)
*Figure 1: Illustration of EEG system.*

This project's objective delves into the domain of neuroscience where we have to parse, filter, preprocess EEG data and train a brain computer interface (BCI) program using the data. A BCI is a 
direct communication link between the brain's electrical activity and an external device, most commonly a computer or robotic limb. See [BCI Wikipedia](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface).

This project builds on machine learning methodologies from previous projects and focus specifically on the following topics:
 - Data parsing and preprocessing using [MNE Python](https://mne.tools/stable/generated/mne.Epochs.html) (frequency filtering using Fourier Transform)
 - Data visualization of brain EEG signals.
 - Feature extraction using Independent component analysis (ICA) and noise filtering (eye movements, neck movements)
 - Data Processing Pipeline using [Scikit-Learn Pipeline class](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html)
 - Cross Kfold validation using [Scikit-Learn KFold](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.KFold.html) and usage of GridSearchCV [Scikit-Learn GridSearchCV](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html) for hyperparameter tuning and best fit.
 - Decision Tree Classifier ML model (coded from scratch to emulate behavior of Scikit-Learn [DecisionTreeClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

Highly recommended reading [neuraldatascience.io](https://neuraldatascience.io/intro.html) for the project. Dataset obtained from [physionet.org](https://physionet.org/content/eegmmidb/1.0.0/) 1.9gb with 6 experiment runs across 109 participants.

Classifier achieved average of 80% accuracy on split test set across 109 participants. Final results **100/100**

### Download [raw edf data](https://physionet.org/content/eegmmidb/1.0.0/)

European Data Format (edf) is a file format used to store physiological data, such as EEG signals in binary format. MNE python library is used to parse the data. 

### To preprocess data
```
python preprocessor.py --start <participant 1 - 109> --end <participant 1 - 109> --experiment <1 - 6> --visualize
```
This will run the script to perform frequency filtering (using Fast Fourier Transform from MNE library to convert data from time domain to frequency domain), perform Independent Component Analysis(ICA) to filter for further noise (ie eye blinks/movements) and run a program - AutoReject - which sample test and remove bad epochs. Results will converted back to time series domain and stored as .csv file in `./preprocessed_data` folder. 

![EEG data before filtering](https://github.com/mseong123/total-perspective-vortex/blob/main/images/EEG_prefilter.png)

*Illustration of EEG data pre - filter*

![Filtering for frequencies](https://github.com/mseong123/total-perspective-vortex/blob/main/images/EEG_postfilter.png)

*Illustration of filtering at 1 hz - 40 hz using frequency domain*

![EEG data post filtering](https://github.com/mseong123/total-perspective-vortex/blob/main/images/Filtering.png)

*Illustration of EEG data post - filter. Data looks cleaner*







