# author: kcgarikipati@gmail.com

import numpy as np
import itertools
import utils
import os

def LoadData(data_path, max_classes = 4, max_length=40):
    '''Load data from the txt files in all subfolders of given data_path
       max_length is the maximum length  for acceleroemter of each data sample
    '''

    data, labels = [], []

    subfolder_list = os.listdir(data_path)
    assert len(subfolder_list) > 0, "Contains no subfolders"
    for ix, subfolder in enumerate(subfolder_list):

        subfolder_path = os.path.join(data_path, subfolder)
        files_list = utils.listdir_fullpath(subfolder_path)
        print("Got {} files in {}".format(len(files_list), subfolder_path))

        labels_identity = np.eye(max_classes)
        for idx, filename in enumerate(files_list):

            # label comes from folder name
            label = int(os.path.splitext(os.path.basename(filename))[0])
            with open(filename, 'r') as f:
                content = f.readlines()

            data_trial = []
            num_trial = 0
            for row in content:
                # trials are separated by "\n"
                if row == "\n":
                    # make all trial length the same size
                    if label == 0:
                        # collect first 40 for pickup
                        data.append(list(itertools.chain(*data_trial[:max_length])))
                    else:
                        # collect last 40 for all others
                        data.append(list(itertools.chain(*data_trial[-max_length:])))
                    # print("data trial len = {}".format(len(data_trial)))
                    labels.append(labels_identity[label])
                    data_trial = []
                    num_trial += 1
                else:
                    row_list = row.strip('\n').split('\t')
                    data_trial.append(row_list)

            # the last trial
            # make all trial length the same size
            if label == 0:
                # collect first 40 for pickup
                data.append(list(itertools.chain(*data_trial[:max_length])))
            else:
                data.append(list(itertools.chain(*data_trial[-max_length:])))
            labels.append(labels_identity[label])
            print("loaded {} trials from {}".format(num_trial+1, filename))

    print("\nTotal trials = {} , Min trial length = {}, Max trial length = {}".format(len(labels),
                                              min(utils.convert_len_2Dlist(data))/3,
                                              max(utils.convert_len_2Dlist(data))/3))

    X, y = np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)
    return X, y