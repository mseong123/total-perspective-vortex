'''define parameters for experiments for training and prediction script bci.py'''

# https://physionet.org/content/eegmmidb/1.0.0/

# experiment one - left fist vs right fist
experiment_one = {
    'filename':[3,7,11],
    'label':['T1', 'T2'],
}
# experiment two - imagine left fist vs right fist
experiment_two = {
    'filaname':[4,8,12],
    'label':['T1', 'T2'],
}

# experiment three - both fists vs both feet
experiment_three = {
    'filename':[5,9,13],
    'label':['T1', 'T2'],
}

# experiment four - imagine both fists vs both feet
experiment_four = {
    'filename':[6,10,14],
    'label':['T1', 'T2'],
}
 
# experiment five - rest vs left fist
experiment_five = {
    'filename':[3,7,11],
    'label':['T0', 'T1'],
}

# experiment six - rest vs imagine both feet
experiment_six = {
    'filename':[6,10,14],
    'label':['T0', 'T2'],
}

def get_const(experiment_no:int) -> dict:
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
