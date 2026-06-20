from atomsci.ddm.pipeline.predict_from_model import predict_from_model_file
from atomsci.ddm.utils import rdkit_easy
from time import time
import argparse, os
import pandas as pd

def main(args):
    #run inference

    input_dataset = pd.read_csv(args.prepared_data_file)
    docking_predictions = predict_from_model_file(args.model_file, input_dataset, smiles_col='smiles', is_featurized=True)

    model_uid = args.model_file.split("_")[-1].split(".")[0]

    output_file_path = os.path.join(args.output_dir, f"scored_validation_data_{model_uid}.csv")
    print("File saved here: ", output_file_path)

    docking_predictions.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared_data_file', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--model_file', required=True, type=str)
    args = parser.parse_args()

    print("inference args: ", args)

    main(args)