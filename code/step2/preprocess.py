import os
import pandas as pd
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

def merge():

    parser = ArgumentParser()#add_help=False)

    parser.add_argument(
        "-s", "--scores", type=Path, required=True, help="hidden size for esm embeddings"
    )

    parser.add_argument(
        "-o", "--out", type=Path, required=True, help="hidden size for esm embeddings"
    )

    args = parser.parse_args()

    score_output = args.scores
    train_out = args.out

    args = parser.parse_args()

    fil_list = sorted(os.listdir(f'{score_output}'))
    
    df = pd.read_csv(f'{score_output}/{fil_list[0]}')
    
    for i in range(1, len(fil_list)):
        df = pd.concat([df, pd.read_csv(f'{score_output}/{fil_list[i]}')])
    
    df.to_csv(f'{train_out}/unprocessed.csv', index=False)

if __name__ == "__main__":
    merge()
