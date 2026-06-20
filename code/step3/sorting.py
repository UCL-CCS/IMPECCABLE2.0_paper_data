import sys
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# from mpi4py import MPI
# import csv
# from collections import OrderedDict
# import json
# import time

fil_list = sorted(os.listdir('Sorting_all'))
df = pd.read_csv(f'Sorting_all/{fil_list[0]}')
for fil in tqdm(fil_list[1:]):
    df_new = pd.read_csv(f'Sorting_all/{fil}')
    df = pd.concat([df, df_new])
    df = df.sort_values(by=['score'], ascending=False)[0:10000]
    del(df_new)

df[:10000].to_csv('sorted_data.csv', index=False)
