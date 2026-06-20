############# Module Loading ##############
# tacke future warning, deprecated warning, bla bla
import warnings
import os
# ------------ tackle some noisy warning
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)

'''
TF_CPP_MIN_LOG_LEVEL = 0 to all logs .
TF_CPP_MIN_LOG_LEVEL = 1 to filter out INFO logs
TF_CPP_MIN_LOG_LEVEL = 2 to additionall filter out WARNING
TF_CPP_MIN_LOG_LEVEL = 3 to additionally filter out ERROR.
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from collections import OrderedDict
import csv
import argparse
import numpy as np
#import matplotlib
import pandas as pd
from mpi4py import MPI
#matplotlib.use("Agg")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, text
from clr_callback import *
from smiles_regress_transformer_funcs_large import *
from tensorflow.python.client import device_lib
import json
from smiles_pair_encoders_functions import *
import time
from tqdm import tqdm
import sys
import argparse

# Limit GPU memory growth
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
parser = argparse.ArgumentParser(description='load model')
parser.add_argument("-w", "--weights", type=Path
)
parser.add_argument("-d", "--dataset", type=str
)
args = parser.parse_args()
databases = args.dataset.split(" ")
#print(args.accumulate(args.integers))

#######HyperParamSetting#############
model_weights = args.weights
json_file = 'config_inference.json'
hyper_params = ParamsJson(json_file)

######## Set up MPI #############

comm, size, rank = initialize_mpi()
os.environ['ROCR_VISIBLE_DEVICES'] = str(rank)
######## Load model #############

print(f"rank is {rank}")
#model = tf.keras.models.load_model('saved_model/my_model')
model = ModelArchitecture(hyper_params).call()
model.load_weights(model_weights)
model.summary()
#sys.exit()
sys.stdout.flush()
####### Oranize data files #########

split_files, split_dirs = large_scale_split(hyper_params, databases, size, rank)
print(f"{rank}:{len(split_files)}")
##### Set up tokenizer ########
if hyper_params['tokenization']['tokenizer']['category'] == 'smilespair':
    vocab_file = hyper_params['tokenization']['tokenizer']['vocab_file']
    spe_file = hyper_params['tokenization']['tokenizer']['spe_file']
    tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)
print(f"{rank}: tokenizer set")
sys.stdout.flush()
#bad_toks = [23, 49, 50, 68, 83, 113, 115, 126, 2083]
####### Iterate over files ##############
BATCH = hyper_params['general']['batch_size']
cutoff = 0
start_total = time.time()

for fil, dirs in tqdm(zip(split_files, split_dirs)):

    if True:
        try:
            Data_smiles_inf, x_inference = large_inference_data_gen(hyper_params,
    tokenizer,
    dirs,
    fil,
    rank)
            
            print(f"{rank}: inference set")
            
            #if any(x in x_inference for x in bad_toks):
            #    continue
            #Data_smiles_inf_split = np.array_split(x_inference, 4)
            Output = model.predict(x_inference,
                                    batch_size = BATCH,
                                    verbose=0)
            
            '''
            Combine SMILES and predicted docking score.
            Sort the data based on the docking score,
            remove data below cutoff score.
            write data to file in output directory
            '''
            SMILES_DS = np.vstack((Data_smiles_inf, np.array(Output).flatten())).T
            SMILES_DS = sorted(SMILES_DS, key=lambda x: float(x[1]), reverse=True)

            #print(SMILES_DS)
            filtered_data = list(OrderedDict((item[0], item) for item in list(SMILES_DS) if float(item[1]) >= cutoff).values())
            filename = f'output/{dirs}/{os.path.splitext(fil)[0]}.dat'
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['smiles', 'score'])
                writer.writerows(filtered_data)

            del (Data_smiles_inf)
            del(Output)
            del(x_inference)
            del(SMILES_DS)
            del(filtered_data)
        except:
            continue

print(f"Started sorting on rank: {rank}")
'''
Sorting per rank
'''

#DATA_FILE_PATH = "output"
#databases = hyper_params['inference_data']['databases']
#All_Files = np.array([])
#All_Dirs = np.array([])
#for dirs in databases:
#    list_dir_files = np.array(sorted(os.listdir(f'{DATA_FILE_PATH}/{dirs}')))
#    All_Files = np.concatenate((All_Files, list_dir_files))
#    dir_enumerate = np.array([dirs for i in range(len(list_dir_files))]) 
#    All_Dirs = np.concatenate((All_Dirs, dir_enumerate))
#
#split_files = np.array_split(All_Files, int(size))[int(rank)]
#split_dirs = np.array_split(All_Dirs, int(size))[int(rank)]

####### Iterate over files ##############

start_total = time.time()

#Sorted_data = []

'''
Sorting all files
parallel merge sort
'''
if True:
        print(rank)
        sys.stdout.flush()
        Sorted_data = pd.DataFrame(columns = ['smiles', 'score'])
        
        it = -1
        from tqdm import tqdm
        for fil, dirs in zip(split_files, split_dirs):
            try:
                it+=1
                filename = f'output/{dirs}/{os.path.splitext(fil)[0]}.dat'
                if it%1000==0:
                    print(it)
                sys.stdout.flush()
                df = pd.read_csv(filename)
                Sorted_data = pd.concat([Sorted_data, df])
                Sorted_data = Sorted_data.drop_duplicates(subset=['smiles'])
                Sorted_data = Sorted_data.sort_values(by=['score'], ascending=False)[0:10000]
                if it%1000==0:
                    print(it)
            except:
                continue

        Sorted_data = Sorted_data.to_numpy()
        Sorted_data = sorted(Sorted_data, key=lambda x: x[1], reverse=True)[0:10000]
        filename = f'Sorting_all/sorted_{rank}.dat'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['smiles', 'score'])
            writer.writerows(Sorted_data)





