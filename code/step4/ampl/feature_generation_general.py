from numba import jit 
from scipy.spatial.distance import cdist 
from multiprocessing import Pool 
import os 
import argparse 
import numpy as np 
import pandas as pd 
import MDAnalysis as mda 
import os, time 
import traceback 
from collections import Counter

    
def collect_file_paths(args):
    master_dict = {} 

    # DCD Files 
    if not os.path.isdir(args.dcd_dir): 
        raise Exception("Directory location for dcd files doesn't exist")

    list_of_files = os.listdir(args.dcd_dir)
    list_of_files.sort()
    for x in list_of_files:
        lig_num = x.split('.')[1]
        path = os.path.join(args.dcd_dir, x)
        if os.path.isfile(path) is False:
            master_dict.pop(x, None)
        else:
            master_dict[lig_num] = [path]

    # PDB Files 
    if not os.path.isdir(args.pdb_dir): 
        raise Exception("Directory location for pdb files doesn't exist") 
    
    list_of_files = os.listdir(args.pdb_dir) 
    list_of_files.sort() 
    for x in list_of_files: 
        lig_num = x.split('.')[1] 
        path = os.path.join(args.pdb_dir, x) 
        if os.path.isfile(path) is False:
            master_dict.pop(x, None)
        else:
            master_dict[lig_num].append(path)

    if args.mode == "training":
        # MMPBSA Files
        if not os.path.isdir(args.mmpbsa_dir):
            raise Exception("Directory location for pdb files doesn't exist")
        list_of_files = os.listdir(args.mmpbsa_dir)
        list_of_files.sort()
        for x in list_of_files:
            path = os.path.join(args.mmpbsa_dir, x)
            path = os.path.join(path, "dg_poses.dat")
            if os.path.isfile(path) is False:
                master_dict.pop(x, None)
            else:
                master_dict[x].append(path)

    # SMILES File 
    with open(args.smiles_file, "r") as f: 
        for line in f: 
            smi = line.split()[0] 
            key = int(line.split()[1])
            if str(key) in master_dict.keys(): 
                master_dict[str(key)].append(smi) 

    output_files = [] 
    for x in master_dict:
        if args.mode == 'training' and len(master_dict[x]) == 4:
            output_files.append(master_dict[x])
        elif args.mode == 'inference' and len(master_dict[x]) == 3:
            output_files.append(master_dict[x])
    print("len(output_files) ", len(output_files))

    chunked_files = [output_files[x:x + args.batch_size] for x in range(0, len(output_files), args.batch_size)]
    return chunked_files 
        

@jit(nopython=True) 
def bucket_features(distances, num_buckets): 
    # Filter out distances so no distances are considered if they are past 12 angstrom 
    cutoff_dist = 12 
    distances = distances[distances <= cutoff_dist]   
    # Generate our bucketed features 
    bucketed_features = [] 

    # Linear buckets 
    step_size = (cutoff_dist - 0) / num_buckets 
    buckets = [step_size * x for x in range(num_buckets)]
    for i in range(num_buckets): 
        low = buckets[i] 
        high = low + step_size 
        bucketed_features.append(np.logical_and(low <= distances, distances < high).sum()) 

    #non-linear buckets
    """
    buckets = [0.1 * x for x in range(0,int(6/0.1))] + [6 + (0.25 * x)
    for x in range(0,int(6/0.25))] #Non-linear buckets
        for i in range(len(buckets)):
            low = buckets[i]
            if i < len(buckets) - 1:
                high = buckets[i+1]
            else:
                high = (buckets[i] - buckets[i-1]) + low
            bucketed_features.append(np.logical_and(low <= distances,distances < high).sum())
    """
    return bucketed_features 

@jit(nopython=True)
def contact_score(distances):
    c_score = 0
    for dist in distances:
            
        c_score += (((1 / dist) ** 12) - ((1 / dist) ** 6))
    
    return c_score

def compute_contacts_training(output_files): 
    num_buckets = 24
    #num_buckets = 1 #Lennard_jones_features
    G = [] 
    UIDs = [] 
    frames = []
    compound_nums = []
    mmpbsa_list = [] 
    pdb_file_names = [] 
    dcd_file_names = [] 
    final_smiles_list = [] 

    for dcd_file, pdb_file, mmpbsa_file, smiles in output_files: 
        try: 
            # Create mda universe 
            univ = mda.Universe(pdb_file, dcd_file) 

            # For UID:
            lig_uid = pdb_file.split('/')[-1].split('.')[1]  
            atom_names = ["H", "C", "N", "O", "S"] 
            
            mmpbsa_values = [] 

            with open(mmpbsa_file, 'r') as f: 
                mmpbsa_values = f.readlines() 
            mmpbsa_values = [x.split()[0] for x in mmpbsa_values]
            #print("num_poses: ", num_poses, " len(mmpbsa_values): ", len(mmpbsa_values))

            if len(univ.trajectory) <= len(mmpbsa_values):
                num_poses = len(univ.trajectory)
            else:
                num_poses = len(mmpbsa_values)

            prot_masks = {} 
            lig_masks = {} 
            for atom in atom_names: 
                prot_mask = univ.select_atoms(f"protein and (name {atom}*)") 
                prot_masks[atom] = prot_mask 
                lig_mask = univ.select_atoms(f"(not protein) and (name {atom}*)") 
                lig_masks[atom] = lig_mask 

            for i in range(num_poses): 
                lig_coords = [] 
                prot_coords = [] 
                # Compute Ligand Coordinates 
                for atom in atom_names: 
                    prot_coords.append(univ.trajectory[i].positions[prot_masks[atom].ix])   
                    lig_coords.append(univ.trajectory[i].positions[lig_masks[atom].ix]) 

                distances = [] 
                for k in range(len(atom_names)): 
                    for l in range(len(atom_names)): 
                        if (prot_coords[k].size == 0) or  (lig_coords[l].size == 0): 
                            distances.append([]) 
                        else: 
                            distances.append(cdist(prot_coords[k], lig_coords[l]).flatten()) 

                # Compute contact scores 
                features = [] 
                for o in range(25): 
                    if len(distances[o]) == 0: 
                        c_score = [0 for _ in range(num_buckets)] 
                        #c_score = 0 #Lennard_jones_features
                    else: 
                        c_score = bucket_features(distances[o], num_buckets) 
                        #c_score = contact_score(distances[o]) #Lennard_jones_features
                    features.extend(c_score) 
                    #features.append(c_score) #Lennard_jones_features
                
                mmpbsa_list.append(mmpbsa_values[i])
                G.append(features) 
                uid = lig_uid + "_" + str(i) 
                UIDs.append(uid) 
                compound_nums.append(lig_uid)
                frames.append(i)
                final_smiles_list.append(smiles) 
                pdb_file_names.append(pdb_file) 
                dcd_file_names.append(dcd_file) 
                 
  
        except Exception: 
            print("Exception occurred: ") 
            print(traceback.format_exc()) 
            print("len(mmpbsa_values): ", len(mmpbsa_values), ", num_poses: ", num_poses, "\n")

    G = np.asarray(G) 
    UIDs = np.array(UIDs) 
    mmpbsa_list = np.array(mmpbsa_list) 
    
    pdb_file_names = np.array(pdb_file_names) 
    dcd_file_names = np.array(dcd_file_names) 
    final_smiles_list = np.array(final_smiles_list) 
    df = prep_for_training(G, UIDs, compound_nums, frames, pdb_file_names, dcd_file_names, mmpbsa_list, final_smiles_list) 
    return df 

def prep_for_training(scores, UIDs, compound_nums, frames, pdb_file_names, dcd_file_names, mmpbsa_list, final_smiles_list): 
    """ 
    This section converts our data into a DataFrame containing all the information necessary for inference and data tracking 
    """ 
    # Prepare data into pandas DataFrame 
    column_labels = [i for i in range(scores.shape[1])]  
    df = pd.DataFrame(scores, columns=column_labels) 
    """frames = []
    compound_nums = []
    for uid in UIDs: 
        frames.append(uid.split("_")[-1])
        compound_nums.append(uid.split("_")[0])"""
  
    df['uid'] = UIDs 
    df['compound_num'] = compound_nums
    df['frame'] = frames
    df['mmpbsa'] = mmpbsa_list 
    df['smiles'] = final_smiles_list 
    df['pdb_file_names'] = pdb_file_names 
    df['dcd_file_names'] = dcd_file_names 
    return df 

def compute_contacts_inference(output_files): 
    num_buckets = 24
    #num_buckets = 1 #Lennard_jones_features
    G = [] 
    UIDs = [] 
    frames = []
    compound_nums = []
    pdb_file_names = [] 
    dcd_file_names = [] 
    final_smiles_list = [] 

    for dcd_file, pdb_file, smiles in output_files: 
        try: 
            # Create mda universe 
            univ = mda.Universe(pdb_file, dcd_file) 

            # For UID:
            lig_uid = pdb_file.split('/')[-1].split('.')[1]  

            atom_names = ["H", "C", "N", "O", "S"] 
            
            num_poses = len(univ.trajectory)

            prot_masks = {} 
            lig_masks = {} 
            for atom in atom_names: 
                prot_mask = univ.select_atoms(f"protein and (name {atom}*)") 
                prot_masks[atom] = prot_mask 
                lig_mask = univ.select_atoms(f"(not protein) and (name {atom}*)") 
                lig_masks[atom] = lig_mask 

            for i in range(num_poses): 
                lig_coords = [] 
                prot_coords = [] 
                # Compute Ligand Coordinates 
                for atom in atom_names: 
                    prot_coords.append(univ.trajectory[i].positions[prot_masks[atom].ix])   
                    lig_coords.append(univ.trajectory[i].positions[lig_masks[atom].ix]) 

                distances = [] 
                for k in range(len(atom_names)): 
                    for l in range(len(atom_names)): 
                        if (prot_coords[k].size == 0) or  (lig_coords[l].size == 0): 
                            distances.append([]) 
                        else: 
                            distances.append(cdist(prot_coords[k], lig_coords[l]).flatten()) 

                # Compute contact scores 
                features = [] 
                for o in range(25): 
                    if len(distances[o]) == 0: 
                        c_score = [0 for _ in range(num_buckets)] 
                        #c_score = 0 #Lennard_jones_features
                    else: 
                        c_score = bucket_features(distances[o], num_buckets) 
                        #c_score = contact_score(distances[o]) #Lennard_jones_features
                    features.extend(c_score) 
                    #features.append(c_score) #Lennard_jones_features
                
                G.append(features) 
                uid = lig_uid + "_" + str(i)
                UIDs.append(uid) 
                compound_nums.append(lig_uid)
                frames.append(i)
                final_smiles_list.append(smiles) 
                pdb_file_names.append(pdb_file) 
                dcd_file_names.append(dcd_file) 
                 
  
        except Exception: 
            print("Exception occurred: ") 
            print(traceback.format_exc()) 

    G = np.asarray(G) 
    UIDs = np.array(UIDs)
    
    pdb_file_names = np.array(pdb_file_names) 
    dcd_file_names = np.array(dcd_file_names) 
    final_smiles_list = np.array(final_smiles_list) 
    df = prep_for_inference(G, UIDs, compound_nums, frames, pdb_file_names, dcd_file_names, final_smiles_list) 
    return df 

def prep_for_inference(scores, UIDs, compound_nums, frames, pdb_file_names, dcd_file_names, final_smiles_list): 
    """ 
    This section converts our data into a DataFrame containing all the information necessary for inference and data tracking 
    """ 
    # Prepare data into pandas DataFrame 
    column_labels = [i for i in range(scores.shape[1])]  
    df = pd.DataFrame(scores, columns=column_labels) 
    """frames = []
    compound_nums = []"""

    

    uid_counter = Counter(UIDs)
    keys_greater_than_1 = [key for key, value in uid_counter.items() if value > 1]
    if len(keys_greater_than_1) > 1:
        print("Number of repeated keys: ", len(keys_greater_than_1))
    else:
        print("All clear")

    """for uid in UIDs: 
        frames.append(uid.split("_")[-1])
        compound_nums.append(uid.split("_")[0])"""
  
    df['uid'] = UIDs 
    #df['uid'] = df['uid'].astype(str) 
    df['compound_num'] = compound_nums
    df['frame'] = frames
    df['smiles'] = final_smiles_list 
    df['pdb_file_names'] = pdb_file_names 
    df['dcd_file_names'] = dcd_file_names 

    print("number of unique id numbers: ", len(df['uid'].unique()), " number of total id numbers: ", len(df['uid']))
    return df 

def main(args): 
    output_files = collect_file_paths(args)

    with Pool(processes=args.num_processes) as pool: 
        if args.mode == 'training':
            out = pool.map(compute_contacts_training, output_files)
        elif args.mode == 'inference':
            out = pool.map(compute_contacts_inference, output_files)
    df = pd.concat(out, axis=0)

    print("After all files processed: ")
    count_dict = Counter(df['uid'])
    keys_greater_than_1 = [key for key, value in count_dict.items() if value > 1]
    if len(keys_greater_than_1) > 1:
        print("Number of repeated keys: ", len(keys_greater_than_1))
    else:
        print("All clear")

    print("number of unique id numbers: ", len(df['uid'].unique()), " number of total id numbers: ", len(df['uid']))


    print("df.shape: ", df.shape)
    df.to_csv(args.prepared_data_file, index=False)

    df2 = pd.read_csv(args.prepared_data_file)

    print("After saving the file and reading it back in:")

    print("number of unique id numbers: ", len(df2['uid'].unique()), " number of total id numbers: ", len(df2['uid']))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    #The ligands are split up into batches. Each batch is then processed and a df is created. After all poses are processed,
    # One pandas df is created then saved to a csv file to be used for inference using AMPL
    parser.add_argument("--batch_size", required=False, default=10, type=int)
    #A directory path containing the pdb files
    parser.add_argument("--pdb_dir", required=True, type=str)
    #A directory path containing the dcd files
    parser.add_argument("--dcd_dir", required=True, type=str)
    #A directory path containing the mmpbsa files
    parser.add_argument("--mmpbsa_dir", required=False, type=str)
    #A file path containing the smiles strings for each of the ligands
    parser.add_argument("--smiles_file", required=True, type=str)
    #A file path to be used by this code. This file path is where the temporary file containing the featurized data will be saved.
    # The pose_selection.py file will then access this csv file to pass into AMPL for inference/score prediction
    parser.add_argument("--prepared_data_file", required=False, default="featurized_data_file.csv", type=str)
    #The number of workers to be spawned to generate the features
    parser.add_argument("--num_processes", required=False, type=int, default=32)
    #Flag to distinguish between training and inference mode
    parser.add_argument("--mode", required=True, type=str)

    args = parser.parse_args()

    assert args.mode in ['training', 'inference']

    if args.mode == 'training':
        assert args.mmpbsa_dir

    print("feature gen args: ", args)

    t1 = time.time()
    main(args)
    t2 = time.time()
    print("Run time: ", t2 - t1)
