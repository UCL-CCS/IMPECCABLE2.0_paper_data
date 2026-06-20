import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "-d", "--batch_size", required=True, help="batch_size"
)
args = parser.parse_args()
batch = int(args.batch_size) 

try:
    df = pd.read_csv('reinvent.csv')
    if len(df) == int(batch):
        print("OK")
    else:
        print("Not OK")

except:
    print("Not OK")
