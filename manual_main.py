from __future__ import print_function

# import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

# %%
from __future__ import print_function
import pdb
import os
import math
# internal imports

from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
# pytorch imports

import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# %%
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# %%
seed_torch()

# %%
encoding_size = 512

# %%
settings = {'num_splits': 1, 
            'k_start': 1,
            'k_end': 1,
            'task': 'task_1_tumor_vs_normal',
            'max_epochs': 200, 
            'results_dir': './results', 
            'lr': 2e-4,
            'experiment': 'crohn_laboratory',
            'reg': 1e-5,
            'label_frac': 1.0,
            'bag_loss': 'ce',
            'seed': 1,
            'model_type': 'clam_sb',
            'model_size': 'small',
            "use_drop_out": False,
            'weighted_sample': False,
            'opt': 'adam'}

# %%
if args['model_type'] in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': 0.7,
                    'inst_loss': 'svm',
                    'B': 1})


# %%
print('\nLoad Dataset')

# %%
if settings['task'] == 'task_1_tumor_vs_normal':
    settings['n_classes']=2
    dataset = Generic_MIL_Dataset(csv_path = '/home/jsy/2023_retro_time_adaptive/data/model_results/test.py/tensor_id/total/label.csv',
                            data_dir= '/home/jsy/2023_retro_time_adaptive/data/model_results/test.py/tensor_id/total',
                            shuffle = False, 
                            seed = settings['seed'], 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=False,
                            ignore=[])

# %%
if not os.path.isdir(settings['results_dir']):
    os.mkdir(settings['results_dir'])

# %%
settings['results_dir'] = os.path.join(settings['results_dir'], str(settings['experiment']) + '_s{}'.format(settings['seed']))

# %%
if not os.path.isdir(settings['results_dir']):
    os.mkdir(settings['results_dir'])

# %%
split_dir = '/home/jsy/2023_clam/CLAM/splits/crohn_laboratory/'

# %%
if split_dir is None:
    split_dir = os.path.join('splits', settings['task']+'_{}'.format(int(settings['label_frac']*100)))
else:
    split_dir = os.path.join('splits', split_dir)

# %%
split_dir

# %%
with open(settings['results_dir'] + '/experiment_{}.txt'.format(settings['experiment']), 'w') as f:
    print(settings, file=f)
f.close()

# %%
print("################# Settings ###################")

# %%
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

# %%
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

# %%
if not os.path.isdir(settings['results_dir']):
        os.mkdir(settings['results_dir'])


# %%
if settings['k_start'] == -1:
    start = 0
else:
    start = settings['k_start']
if settings['k_end'] == -1:
    end = settings['k']
else:
    end = settings['k_end']

# %%
all_test_auc = []
all_val_auc = []
all_test_acc = []
all_val_acc = []
folds = np.arange(start, end)

# %%
from argparse import Namespace

# %%

args = Namespace(
    num_splits=1,
    k_start=1,
    k_end=1,
    task='task_1_tumor_vs_normal',
    max_epochs=200,
    results_dir='./results',
    lr=2e-4,
    experiment='crohn_laboratory',
    reg=1e-5,
    label_frac=1.0,
    bag_loss='ce',
    seed=1,
    model_type='clam_sb',
    model_size='small',
    use_drop_out=False,
    weighted_sample=False,
    opt='adam'
)

# %%
for i in folds:
    seed_torch(settings['seed'])
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
            csv_path='{}/splits_{}.csv'.format(settings['split_dir'], i))

    datasets = (train_dataset, val_dataset, test_dataset)
    results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    #write results to pkl
    filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
    save_pkl(filename, results)


# %%
args = Namespace(
    num_splits=1,
    k_start=1,
    k_end=1,
    k =1,
    task='task_1_tumor_vs_normal',
    max_epochs=200,
    results_dir='./results',
    lr=2e-4,
    experiment='crohn_laboratory',
    reg=1e-5,
    label_frac=1.0,
    bag_loss='ce',
    seed=1,
    model_type='clam_sb',
    model_size='small',
    use_drop_out=False,
    weighted_sample=False,
    opt='adam'
)

# %%
final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

if len(folds) != args.k:
    save_name = 'summary_partial_{}_{}.csv'.format(start, end)
else:
    save_name = 'summary.csv'
final_df.to_csv(os.path.join(args.results_dir, save_name))

# %%
final_df

# %%
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
results = main(args)

# %%
def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# %%
results = main(args)


