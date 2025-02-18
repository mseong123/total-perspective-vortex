# total-perspective-vortex | 42KL


![Neural Network](https://github.com/mseong123/total-perspective-vortex/blob/main/images/EEG_system.jpg)
*Figure 1: Illustration of EEG system.*

This project's objective delves into the domain of neuroscience where we have to parse, filter, preprocess EEG data and train a brain computer interface (BCI) program using the data. A BCI is a 
direct communication link between the brain's electrical activity and an external device, most commonly a computer or robotic limb. See [BCI Wikipedia](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface).

This project builds on machine learning methodologies from previous projects and focus specifically on the following topics:
 - Data parsing and preprocessing using [MNE Python](https://mne.tools/stable/generated/mne.Epochs.html) (frequency filtering using Fourier Transform)
 - Data visualization of brain EEG signals.
 - Feature extraction using Independent component analysis (ICA) and noise filtering (eye movements, neck movements).
 - Data Processing Pipeline using [Scikit-Learn Pipeline class](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).
 - Dimensionality reduction using Principal Component Analysis (PCA).
 - Cross Kfold validation using [Scikit-Learn KFold](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.KFold.html) and usage of GridSearchCV [Scikit-Learn GridSearchCV](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html) for hyperparameter tuning and best fit.
 - Implementation of my own Decision Tree Classifier ML model (coded from scratch to emulate behavior of Scikit-Learn [DecisionTreeClassifier](https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

Highly recommended reading [neuraldatascience.io](https://neuraldatascience.io/intro.html) for the project. Dataset obtained from [physionet.org](https://physionet.org/content/eegmmidb/1.0.0/) 1.9gb with 6 experiment runs across 109 participants.

Classifier achieved average of 84% accuracy on split test set across 109 participants. Final results **100/100**

**TLDR** Very challenging project due to required domain expertise in parsing signals. Also the methodology of dataset split to train and test set is problematic. Each experiment run has 19500 time series data.
In theory, model should generalise well and able to predict on a fully independent test set that consist of one full segregated experiment run for each participant). However accuracy of my models are quite low hence i have to sum my datapoints together in all experiment runs and split them. It could be due to inadequate feature engineering on the EEG dataset (not using ICA, CSP or CSSP) to better transform the dataset
into their independent components so that the model can learn better. 

### Download [raw edf data](https://physionet.org/content/eegmmidb/1.0.0/)

European Data Format (edf) is a file format used to store physiological data, such as EEG signals in binary format. MNE python library is used to parse the data. 

### To preprocess data
```
python preprocessor.py --start <participant 1 - 109> --end <participant 1 - 109> --experiment <1 - 6> --visualize
```
This will run the script to perform frequency filtering (using Fast Fourier Transform from MNE library to convert data from time domain to frequency domain), perform Independent Component Analysis(ICA) to filter for further noise (ie eye blinks/movements) and run a program - AutoReject - which sample test and remove bad epochs. Results will converted back to time series domain and stored as .csv file in `./preprocessed_data` folder. 

![EEG data before filtering](https://github.com/mseong123/total-perspective-vortex/blob/main/images/EEG_prefilter.png)

*Illustration of EEG data pre - filter*

![Filtering for frequencies](https://github.com/mseong123/total-perspective-vortex/blob/main/images/Filtering.png)

*Illustration of filtering at 1 hz - 40 hz using frequency domain. Human brain waves for motor imaginery movements typically at 8hz to 40hz.*

![Human brain waves frequency](https://github.com/mseong123/total-perspective-vortex/blob/main/images/human_brain_waves_frequency.png)


![EEG data post filtering](https://github.com/mseong123/total-perspective-vortex/blob/main/images/EEG_postfilter.png)

*Illustration of EEG data post - filter. Data looks 'cleaner'*

### To train model
```
python bci.py train --experiment <1 - 6> --subject <1 - 109>
```
This will run the script to train subject and experiment (using 70/30 split between `training` and `test` data) using Scikit-Learn data Pipeline object (which includes further processing using StandardScaler and my own implementation of PCA to reduce number of features while retaining amount of information in dataset). Training process also includes Cross KFold validation for more consistent score and GridSearchCV for hyperparameter tuning. Ensure that associated .csv preprocessed file is preprocessed first (above) and stored in `./preprocessed_data` folder. Cross KFold scores and average accuracy is shown at end of training and Python object (fitted model) is stored in `./model` folder.

### To predict model
```
python bci.py train --experiment <1 - 6> --subject <1 - 109>
```

This will run the script to stream `test` data at 1000 samples per second and outputs are displayed in terminal. Ensure that associated model(.pkl file) is trained first (above) and stored in `./model` folder.

### To train all
```
python bci.py
```
This will run script to train all 6 experiments for each 109 participants (654 in total) and output the scores on their respective test set. Results are stored in result.txt file. Average score is 84%.


## BONUS implementation

Coded my own implementation of DecisionTreeClassifier (decision_tree.py) based on the model provided by Scikit-Learn with the following hyperparameters implementation:
 - criterion = 'gini' or 'entropy'
 - max-depth
 - min_samples_split

The algorithm is done recursively and the class instance have a fit(), predict() and score() method as well. Data structure and attributes are stored as close as possible to Scikit-Learn's implementation including the following:
 - classes_
 - n_features_in_
 - tree_
    - children_left
    - children_right
    - feature
    - threshold
    - value
    - impurity
    - n_node_sample

### Test decision tree classifier
```
python decision_tree_test.py
```
Accuracy score is 100% on any sample test set provided.




