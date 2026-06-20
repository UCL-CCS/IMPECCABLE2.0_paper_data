from atomsci.ddm.pipeline.predict_from_model import predict_from_model_file
from atomsci.ddm.utils import rdkit_easy
from time import time
import argparse, os, yaml
import pandas as pd

from sklearn.metrics import r2_score, confusion_matrix, classification_report




def compute_rankings(merged_df):
    """
    The rankings are relative to each smiles string. For each smiles string, each pose is assigned a rank (1, n) 
    where n is the number of poses that smiles string has. There are two rankings that are computed:

    1. mmpbsa_rank: this is computed based on the truth mmpbsa values
    2. mmpbsa_pred_rank: this is computed based on the predicted mmpbsa values

    After the rankings are generated, there are more boolean columns added that contain a 1 if the ranking (truth and predicted)
    is below or equal to some threshold (i.e. one of the top 5 poses) and a 0 otherwise. These columns can be used later 
    in the compute_classification_stats function.
    """

    for smi in merged_df['smiles'].unique().tolist():
        #Set a ranking sorted by the truth mmpbsa value
        merged_df.loc[merged_df['smiles'] == smi, 'mmpbsa_rank'] = merged_df.loc[merged_df['smiles'] == smi, 'mmpbsa'].rank(method='first')

        #Set a ranking sorted by the predicted mmpbsa value
        merged_df.loc[merged_df['smiles'] == smi, 'mmpbsa_pred_rank'] = merged_df.loc[merged_df['smiles'] == smi, 'mmpbsa_pred'].rank(method='first')

    #Add rankings for top 5, 10, and 15
    #Truth values
    merged_df.loc[merged_df['mmpbsa_rank'] > 1,'top_1_truth'] = 0
    merged_df.loc[merged_df['mmpbsa_rank'] == 1,'top_1_truth'] = 1

    merged_df.loc[merged_df['mmpbsa_rank'] > 5,'top_5_truth'] = 0
    merged_df.loc[merged_df['mmpbsa_rank'] <= 5,'top_5_truth'] = 1

    merged_df.loc[merged_df['mmpbsa_rank'] > 10,'top_10_truth'] = 0
    merged_df.loc[merged_df['mmpbsa_rank'] <= 10,'top_10_truth'] = 1

    merged_df.loc[merged_df['mmpbsa_rank'] > 15,'top_15_truth'] = 0
    merged_df.loc[merged_df['mmpbsa_rank'] <= 15,'top_15_truth'] = 1

    merged_df.loc[merged_df['mmpbsa_rank'] > 25,'top_25_truth'] = 0
    merged_df.loc[merged_df['mmpbsa_rank'] <= 25,'top_25_truth'] = 1

    #Predicted values
    merged_df.loc[merged_df['mmpbsa_pred_rank'] > 1,'top_1_predicted'] = 0
    merged_df.loc[merged_df['mmpbsa_pred_rank'] == 1,'top_1_predicted'] = 1

    merged_df.loc[merged_df['mmpbsa_pred_rank'] > 5,'top_5_predicted'] = 0
    merged_df.loc[merged_df['mmpbsa_pred_rank'] <= 5,'top_5_predicted'] = 1

    merged_df.loc[merged_df['mmpbsa_pred_rank'] > 10,'top_10_predicted'] = 0
    merged_df.loc[merged_df['mmpbsa_pred_rank'] <= 10,'top_10_predicted'] = 1

    merged_df.loc[merged_df['mmpbsa_pred_rank'] > 15,'top_15_predicted'] = 0
    merged_df.loc[merged_df['mmpbsa_pred_rank'] <= 15,'top_15_predicted'] = 1

    merged_df.loc[merged_df['mmpbsa_pred_rank'] > 25,'top_25_predicted'] = 0
    merged_df.loc[merged_df['mmpbsa_pred_rank'] <= 25,'top_25_predicted'] = 1

    return merged_df

def compute_classification_stats(merged_df, stats_dict):
    """
    This function generates classification based metrics (sensitivity, specificity, and accuracy) on the boolean columns
    created in the compute_rankings function. These metrics are computed and added to the stats_dict.
    """
    top_1_class_report = classification_report(merged_df['top_1_truth'].tolist(), merged_df['top_1_predicted'].tolist()).split()
    sensitivity, specificity, accuracy = top_1_class_report[11], top_1_class_report[6], top_1_class_report[15]
    stats_dict['top_1_poses'] = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}

    top_5_class_report = classification_report(merged_df['top_5_truth'].tolist(), merged_df['top_5_predicted'].tolist()).split()
    sensitivity, specificity, accuracy = top_5_class_report[11], top_5_class_report[6], top_5_class_report[15]
    stats_dict['top_5_poses'] = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}

    top_10_class_report = classification_report(merged_df['top_10_truth'].tolist(), merged_df['top_10_predicted'].tolist()).split()
    sensitivity, specificity, accuracy = top_10_class_report[11], top_10_class_report[6], top_10_class_report[15]
    stats_dict['top_10_poses'] = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}

    top_15_class_report = classification_report(merged_df['top_15_truth'].tolist(), merged_df['top_15_predicted'].tolist()).split()
    sensitivity, specificity, accuracy = top_15_class_report[11], top_15_class_report[6], top_15_class_report[15]
    stats_dict['top_15_poses'] = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}

    top_25_class_report = classification_report(merged_df['top_25_truth'].tolist(), merged_df['top_25_predicted'].tolist()).split()
    sensitivity, specificity, accuracy = top_25_class_report[11], top_25_class_report[6], top_25_class_report[15]
    stats_dict['top_25_poses'] = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}

    return stats_dict

def compute_confidence(merged_df, stats_dict, goal_percent, top_n_poses):
    """
    This function determines the number (n) that are the top n predicted poses that would need to be selected 
    for at least one of the top_n_poses for goal_percent of the smiles strings.

    merged_df: pandas df
    stats_dict: dict
    goal_percent = float
    top_n_poses = int
    """
    num_poses = 20
    done = False
    nums_visited = []
    current_best = [100, 0, 100]

    num_unique_smiles = merged_df['smiles'].unique().shape[0]

    while not done:
        reduced_df_pred_top_n = merged_df.loc[merged_df['mmpbsa_pred_rank'] <= num_poses]
        reduced_df_pred_top_n_with_truth_top_m = reduced_df_pred_top_n.loc[reduced_df_pred_top_n['mmpbsa_rank'] <= top_n_poses]

        perc_truth_top_m_in_predicted_top_n = reduced_df_pred_top_n_with_truth_top_m['smiles'].unique().shape[0] / num_unique_smiles

        difference_from_goal_percent = goal_percent - perc_truth_top_m_in_predicted_top_n

        if abs(difference_from_goal_percent) < abs(current_best[2]):
            current_best = [num_poses, perc_truth_top_m_in_predicted_top_n, difference_from_goal_percent]

        if num_poses in nums_visited:
            break
        else:
            nums_visited.append(num_poses)
        
        if difference_from_goal_percent == 0:
            break
        elif difference_from_goal_percent > 0:
            num_poses += 1
        else:
            num_poses -= 1
        
    predicted_top_n, percent_smiles = current_best[0], current_best[1]

    stats_dict[f"Number of top predicted poses needed to be pulled to at least contain one of the top {top_n_poses} poses for {goal_percent*100}% of smiles strings"] = predicted_top_n

    return stats_dict

def run_validation(docking_predictions):
    truth_mmpbsa_path = "/lustre/orion/chm155/proj-shared/AMPL/validation_aug2024/lig-pose-mmpbsa.csv"
    mmpbsa_df = pd.read_csv(truth_mmpbsa_path)
    
    #Merge the truth mmpbsa values into the predicted csv file
    merged_df = docking_predictions.merge(mmpbsa_df, on = ['compound_num', 'frame'], how= 'left')
    merged_df = merged_df[merged_df['mmpbsa'].isna() == False]
    
    #Compute the r2 score of the mmpbsa predictions and add it to the stats_dict
    r2 = r2_score(merged_df['mmpbsa'].tolist(), merged_df['mmpbsa_pred'].tolist())
    stats_dict["r2"] = float(r2)
    
    merged_df = compute_rankings(merged_df)
    
    stats_dict = compute_classification_stats(merged_df, stats_dict)

    stats_dict = compute_confidence(merged_df, stats_dict, 0.90, 1)
    stats_dict = compute_confidence(merged_df, stats_dict, 0.90, 3)
    stats_dict = compute_confidence(merged_df, stats_dict, 0.90, 5)

    with open(args.output_dir, 'w') as f:
        yaml.dump(stats_dict, f)



def run_inference(args):
    #run inference

    input_dataset = pd.read_csv(args.prepared_data_file)
    docking_predictions = predict_from_model_file(args.model_file, input_dataset, smiles_col='smiles', is_featurized=True)

    model_uid = args.model_file.split("_")[-1].split(".")[0]

    output_file_path = os.path.join(args.output_dir, f"scored_validation_data_{model_uid}.csv")
    print("File saved here: ", output_file_path)

    docking_predictions.to_csv(output_file_path, index=False)

    return docking_predictions



def main(args):

    docking_predictions = run_inference(args)

    run_validation(docking_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--featurized_validation_data_file", required=True, type=str) #/lustre/orion/chm155/proj-shared/AMPL/paper_writing/molecular_pose/data_files/inference_data_featurized.csv
    parser.add_argument("--model_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    t0 = time.time()
    main(args)
    t1 = time.time()
    
    print(f"Time to run: {t1-t0}")