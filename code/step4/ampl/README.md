# Conda environments

We have two conda environments each stored in the shared python install. The below environment is the conda environment for AMPL version 1.6.0. This environment is used for model training and inference.

`/ccs/proj/chm155/IMPECCABLE/miniconda/envs/atomsci160`

Our second environment is the below path. This environment is used for our data visualization, validation, and feature generation. 

`/ccs/proj/chm155/IMPECCABLE/miniconda/envs/data_prep_env`

Our AMPL repository is installed in the below path. If you need to make any code changes, you can do it here although it isn't recommended to change code within AMPL unless necessary as it can break the pipeline.

`/ccs/proj/chm155/blackst/AMPL-scaling`

If you do make any changes, please make sure to run the following command after to put those changes into effect in the conda environment installation of the code:

`./build.sh && ./install.sh system`

# Input/output

### Inputs from other components

- pdb_dir: This is a file path leading to a directory containing the pdb files for input. Will be provided to us once the docking poses are generated. Our feature generation code is currently written to accept a certain file structure, but can be rewritten once we decide on standards if need be. See (`/lustre/orion/chm155/proj-shared/ESMACS+RCT/tnks2-ML-docking/Model-generation/output_combined_trajectories/pdbs`) for example.

- dcd_dir: This is a file path leading to a directory containing the dcd files for input. Will be provided to us once the docking poses are generated. Our feature generation code is currently written to accept a certain file structure, but can be rewritten once we decide on standards if need be. See (`/lustre/orion/chm155/proj-shared/ESMACS+RCT/tnks2-ML-docking/Model-generation/output_combined_trajectories/dcds`) for example.

- pdb_dir: This is a file path leading to a file containing the SMILES strings for input. Will be provided to us once the docking poses are generated. Our feature generation code is currently written to accept a certain file structure, but can be rewritten once we decide on standards if need be. See (`/lustre/orion/chm155/proj-shared/ESMACS+RCT/tnks2-ML-docking/docking/compounds.smi`) for example.

### Internal inputs

- prepared_data_file: this is a file path needed for our featurized data to be saved off before being ran through inference. Doesn't need to be different every time, but can be if we want to save off our featurized data for future model improvements. (ex. `/lustre/orion/chm155/proj-shared/AMPL/example_data.csv`)

- num_processes: An integer value for the number of workers to be spawned for feature generation.

- number_of_poses: An integer value for the number of poses to be returned for each ligand. The pose_selection.py file will score the datapoints from the prepared_data_file, sort the poses by predicted mmpbsa values for each ligand, and return the n (`number_of_poses`) poses with the best predicted mmpbsa values. Not sure if this will be constant across a run of IMPECCABLE or if this will change of time.

### Outputs

- output_dir: This is a file path directing our code where to save off the proposed poses. Format may change as standards are defined.


# Getting started

- working directory in Slurm-pose_prediction file needs to be modified


# Other notes

- There is a file provided (`/ccs/home/blackst/IMPECCABLE_2.0/ampl_pose_prediction/inference.py`) that will run inference on a provided dataset with the provided pre-trained AMPL model. Use this during testing/training process to run the validation set. It will return the predicted score for every molecule and save that off into a separate csv file.

- There is also a file provided (`/ccs/home/blackst/IMPECCABLE_2.0/ampl_pose_prediction/pose_selection.py`) that is similar but with a key difference. pose_selection.py performs inference the same way that inference.py does. However, the difference is that pose_selection.py is intended to be used in the IMPECCABLE pipeline. The difference being that pose_selection will run inference then remove all but the top number_of_poses (user defined parameter) based on the predicted score. 

- The script to generate features (`/ccs/home/blackst/IMPECCABLE_2.0/ampl_pose_prediction/feature_generation_general.py`) is saved here. This can be run in two modes: training and inference. The difference being that for our training data, we need to merge in the truth MMPBSA values. So if mode is set to training, it will expect an mmpbsa file path to be provided. Inference mode on the other hand does not require the mmpbsa file to be provided as we keep that separate until after running inference. 

- We have set up using IMPECCABLE's framework to do large batch parallel training. To do this, we use the `wfms/rp/modules/ampl.py` and `IMPECCABLE_2.0/ampl_pose_prediction/ampl_config.json` to train multiple AMPL models at the same time. By doing this, we can train multiple models each being given 1 GPU (the max that can be used on any one model currently - limitation on AMPL's side) and different parameters. This allows for HPO to be performed covering a large range of parameter values or different datasets even. If you need help setting this up, please contact Sean Black. 

