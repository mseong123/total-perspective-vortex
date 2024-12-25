'''script to preprocess eeg data and to visualize preprocessing pipeline'''
import mne
import matplotlib.pyplot as plt
import numpy as np
import argparse

def define_args() -> argparse.Namespace:
    """define arguments in command line"""
    parser = argparse.ArgumentParser(description="A preprocessing script to for eeg data preprocessing.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparser_indiv = subparsers.add_parser("individual", help="option for individual eeg files preprocessing split by subject and experiment")
    subparser_indiv.add_argument("subject",default=1, type=int, help="subject no. (1 to 109) e.g. 1, Default:1")
    subparser_indiv.add_argument("experiment",default=3, type=int, help="INT:experiment (1 - 6) e.g. 3, Default:3")
    subparser_indiv.add_argument("--visualize", default = False, action="store_true", help="enable graph visualization e.g. TRUE, Default=False")
    subparser_batch = subparsers.add_parser("batch", help="option for batch eeg files preprocessing split by subject and experiment")
    subparser_batch.add_argument("start",default=1, type=int, help="start of subject no. (1 to 109), e.g. 1, Default:1")
    subparser_batch.add_argument("end",default=109, type=int, help="end of subject no. (1 to 109), e.g. 109, Default:109")
    subparser_batch.add_argument("experiment",default=3, type=int, help="experiment (1 - 6) e.g. 3, Default:3")
    return parser.parse_args()

def main()-> None:
    '''main function for preprocessing'''
    args:argparse.Namespace = define_args()
    print(args.mode)


if __name__ == '__main__':
    main()