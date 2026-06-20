from atomsci.ddm.pipeline.predict_from_model import predict_from_model_file
from atomsci.ddm.utils import rdkit_easy
from time import time
import pandas as pd

def main():

    model_file = "../selected_trained_model.tar.gz"

    # run inference
    input_dataset = pd.read_csv("featurized_data_file.csv")
    docking_predictions = predict_from_model_file(model_file, input_dataset, smiles_col='smiles', is_featurized=True)

    # Return the predicted mmpbsa values in the format used in step 5
    mini_df_list = []
    for x in docking_predictions['compound_num'].unique():
        mini_df = docking_predictions[docking_predictions['compound_num'] == x]
        mini_df = mini_df.sort_values(by="mmpbsa_pred")
        
        mini_df.reset_index(inplace=True)
        mini_df = mini_df[mini_df.index < 50]

        mini_df_list.append(mini_df)


    filtered_df = pd.concat(mini_df_list, ignore_index=True)
    filtered_df.to_csv(f"filtered_poses.csv", index=False)

if __name__ == "__main__":

    main()