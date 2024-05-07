import pandas as pd
import numpy as np
import argparse
import torch
import lib
import os

parser = argparse.ArgumentParser()
parser.add_argument("-seed", type=int, default=22, help="Seed for random initialization") #Random seed setting
parser.add_argument('--data_folder', default='./data_after/', type=str)
parser.add_argument('--train_data', default='recSys15TrainOnly.txt', type=str)
parser.add_argument('--valid_data', default='recSys15Valid.txt', type=str)
parser.add_argument('--test_data',default='recSys15Test.txt',type=str)

# Get the arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
#use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def subsample(path, type, ratio):
    # Load the original test data
    original_test_data = pd.read_csv(path)

    # Get unique session IDs
    unique_sessions = original_test_data['SessionID'].unique()
    np.random.shuffle(unique_sessions)

    selected_sessions = unique_sessions[:ratio]
    filtered_test_data = original_test_data[original_test_data['SessionID'].isin(selected_sessions)]
    print('Filtered {} Set has'.format(type), len(filtered_test_data), 'Events,', filtered_test_data['SessionID'].nunique(), 'Sessions, and', filtered_test_data['ItemID'].nunique(), 'Items\n\n')

    # Save the filtered test data to a new file
    filtered_test_data.to_csv('recSys{}_{}.txt'.format(type,ratio), sep=',', index=False)

def main():
    subsample(os.path.join(args.data_folder, args.train_data), "Train", 1000)
    subsample(os.path.join(args.data_folder, args.valid_data), "Valid", 10)
    subsample(os.path.join(args.data_folder, args.test_data), "Test", 300)


if __name__ == '__main__':
    main()