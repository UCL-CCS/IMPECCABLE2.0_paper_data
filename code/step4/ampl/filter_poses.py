import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter
import os
import pandas as pd

def main():
    filtered_df = pd.read_csv("./filtered_poses.csv")

    for compound in filtered_df['compound_num'].unique():
        try:
            pdb_path = f"./output_combined_trajectories/pdbs/rec_4ui5.{compound}.dcd"
            dcd_path = f"./output_combined_trajectories/dcds/rec_4ui5.{compound}.dcd"

            selected_frames = filtered_df.loc[filtered_df["compound_num"] == compound, "frame"].tolist()

            u = mda.Universe(pdb_path, dcd_path)

            subset_dcd = f"./output_combined_trajectories/dcds/rec_4ui5.{compound}_sorted.dcd"

            # Write selected frames to a new DCD file
            with DCDWriter(subset_dcd, n_atoms=u.atoms.n_atoms) as dcd:
                for frame in selected_frames:
                    u.trajectory[frame]  # Move to the selected frame
                    dcd.write(u)  # Write the frame
        
        except Exception as e:
            print("Exception: ", e)
            continue

if __name__ == "__main__":
    main()