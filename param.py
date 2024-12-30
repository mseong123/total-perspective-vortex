'''define parameters for experiments for training and prediction script bci.py'''
# data source
# https://physionet.org/content/eegmmidb/1.0.0/

# experiment one - left fist vs right fist
experiment_one = {
    'file_no':[3, 7, 11],
    'label':['T1', 'T2'],
}
# experiment two - imagine left fist vs right fist
experiment_two = {
    'fila_no':[4, 8, 12],
    'label':['T1', 'T2'],
}

# experiment three - both fists vs both feet
experiment_three = {
    'file_no':[5, 9, 13],
    'label':['T1', 'T2'],
}

# experiment four - imagine both fists vs both feet
experiment_four = {
    'file_no':[6, 10, 14],
    'label':['T1', 'T2'],
}
 
# experiment five - rest vs left fist
experiment_five = {
    'file_no':[3, 7, 11],
    'label':['T0', 'T1'],
}

# experiment six - rest vs imagine both feet
experiment_six = {
    'file_no':[6, 10, 14],
    'label':['T0', 'T2'],
}
# constant params
RANDOM_STATE = 42
PREPROCESSED_PATH = './preprocessed_data/'
DATA_PATH = './data/files/'
MODEL_PATH = './model/'
MEMORY_CACHE_PATH = './memcache'
TEST_SIZE = 0.3
BATCH_SIZE = 100

def get_param(experiment_no:int) -> dict:
    '''return dict of paramaters for relevant experiment'''
    if experiment_no == 1:
        return experiment_one
    elif experiment_no == 2:
        return experiment_two
    elif experiment_no == 3:
        return experiment_three
    elif experiment_no == 4:
        return experiment_four
    elif experiment_no == 5:
        return experiment_five
    elif experiment_no == 6:
        return experiment_six


def get_prefix(i:int)->str:
    if i < 10:
        return "00"
    elif i < 100:
        return "0"
    else:
        return ""