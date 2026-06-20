import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='input file')
parser.add_argument("-i", "--inp")

args = parser.parse_args()


df = pd.DataFrame()

columns_not_done = ['Agent', 'Prior', 'Target', 'Score', 'SMILES', 'Scaffold', 'QED', 'QED (raw)', 'Stereo', 'Stereo (raw)', 'ChemProp', 'ChemProp (raw)', 'Alerts', 'Alerts (raw)', 'step', 'cluster']
columns_string = ['SMILES', 'Scaffold']
columns_done = ['OE_SMILES', 'dG_ESMACS', 'sd_ESMACS']

data_done = pd.read_csv(args.inp, names=columns_done)
df_len = len(data_done)
for c in columns_not_done:
    if c not in columns_string:
        df[c]=['' for it in range(df_len)]
    elif c in columns_string:
        df[c]=['' for it in range(df_len)]
for c in columns_done:
    df[c] = data_done[c]

df.to_csv('esmacs.csv', index=False)
