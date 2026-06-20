import argparse, json

import atomsci.ddm.pipeline.model_pipeline as mp
import atomsci.ddm.pipeline.parameter_parser as parse


def get_model_training_params(args):
    with open("training_params.json", 'r') as f:
        full_parameters = json.load(f)
    
    return full_parameters[str(args.index)]

def train(args):

    variable_params = get_model_training_params(args)

    bs, ls, lr, wd = variable_params

    params = {
        "dataset_key":          "./featurized_data_file.csv", 
        "datastore":            False,
        "splitter":             "random",
        "split_valid_frac":     "0.15",
        "split_test_frac":      "0.15",
        "split_strategy":       "train_valid_test", 
        "prediction_type":      "regression",
        "response_cols":        "mmpbsa",
        "id_col":               "uid",
        "smiles_col":           "smiles",
        "result_dir":           "./model_store",
        "model_type":           "NN",
        "featurizer":           "computed_descriptors",
        "descriptor_type":      "bin_cont_scores_24",
        "previously_featurized":    True,
        #"descriptor_type":      args.descriptor_type,
        "max_epochs":           40,
        "weight_decay_penalty_type": wd,
        "learning_rate":        lr,
        "layer_size":           ls,
        "batch_size":           bs
    }

    pparams = parse.wrapper(params)
    MP = mp.ModelPipeline(pparams)
    
    MP.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--learning_rate", required=False, default=0.05, type=float)
    #parser.add_argument("--train_batch_size", required=False, default=1000, type=int)
    #parser.add_argument("--layer_size", required=True, type=str)
    #parser.add_argument("--weight_decay_penalty_type", required=True, type=str)
    #parser.add_argument("--result_dir", required=False, default="./model_store" type=str)
    #parser.add_argument("--featurized_data_file.csv", required=False, type=str)
    #parser.add_argument("--descriptor_type", required=False, default="bin_cont_scores_24", type=str)
    parser.add_argument("--index", required=True, type=int) 
    args = parser.parse_args()

    output = train(args)