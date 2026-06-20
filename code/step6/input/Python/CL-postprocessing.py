"""Prepare REINVENT generated SMILES for ESMACS with a defined batch size

REINVENT generates a CSV file with information of generated SMILES and
associated scores.  This script reads single or multiple such CSV files.
The data is filtered by ESNACS dG and QEDs, rows with duplicate SMILES are
dropped.

The SMILES can be optinally clustered using RDKit fingerprint and Tanimoto
distance.  Each cluster is pruned by retaining only the top N, by dG, SMILES.
Drops SMILES with the highest dG in each cluster to adjust the data size to
the desired batch size.
"""

import os
import sys
import argparse
import logging
import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning
import umap # load before NUMBA/numpy
import numpy as np
import hdbscan
#from sklearn.cluster import DBSCAN
#from bk_clustering import BurjKhalifaClustering
from rdkit import Chem


__progname__ = "CL-postprocessing"
__version__ = "0.0.0"
__copyright__ = "(C) AstraZeneca 2024"
__authors__ = "Hannes H Loeffler"

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt="[%Y-%m-%d %H:%M:%S %Z]", level=logging.INFO)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"{__progname__}: postprocess CSV file(s) from a REINVENT "
        "CL run.  The SMILES will be collected, filtered and (optionally) "
        "clustered.  A final set of SMILES is written to a new CSV file.",
        epilog=f"v{__version__} {__copyright__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "csv_filenames",
        nargs="+",
        default=None,
        metavar="FILE",
        type=os.path.abspath,
        help="Input CSV file(s) from a REINVENT CL run",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="N",
        type=int,
        default=500,
        help="Number of compounds for the batch to write out.  Approximate if "
             "clustering is requested.  In that case the number will be "
             "slightly larger.",
    )

    parser.add_argument(
        "-o",
        "--output-csv",
        metavar="FILE",
        default="batch.csv",
        help="Output CSV for the new batch with the selected SMILES.",
    )

    parser.add_argument(
        "-i",
        "--input-column",
        metavar="NAME",
        default="SMILES",
        help="Column header for the SMILES in the input CSV file.",
    )

    parser.add_argument(
        "-r",
        "--response-column",
        metavar="NAME",
        default="ChemProp (raw)",
        help="Column header for the response in the input CSV file.",
    )

    parser.add_argument(
        "-n",
        "--no-clustering",
        action='store_true',
        help=f"Do not cluster the REINVENT compounds.  The top N (by dG) "
             "will be selected, for N see batch size parameter.",
    )

    return parser.parse_args()


QED_LABEL = "QED"

def filter_df(df, response_label):
    """Filter the dataframe.

    Somwehat arbitrary filtering by QED and REINVENT predicted dG.  May need
    adjusting.

    :param df: Pandas dataframe
    """

    QED_min = df[QED_LABEL] > 0.6
    dG_max = df[response_label] < -10.0
    good_score = df['Score'] > 0
    all_good = QED_min & dG_max & good_score

    if "Alerts" in df.columns:
        good_mols = df["Alerts"] > 0
        all_good &= good_mols

    return df[all_good]


def create_UMAP(fps):
    """Compute the UMAP from fingerprints

    May need some fine-tuning of the hyper parameters.

    :param fps: molecule fingerprints
    :returns: UMAP data
    """

    umap_model = umap.UMAP(
        n_components = 2,  # number of dimensions
        metric = "jaccard",
        n_neighbors = 50,  # if larger then more focus on overall structure
        min_dist = 0.001,  # if larger then structures pushed nore apart into softer, more general features
        low_memory = False
    )

    return umap_model.fit_transform(fps)


def cluster_hdbscan(X):
    """Density bnased HDBSCAN clustering.

    May need some fine-tuning of the hyper parameters.

    :param X: data to cluster
    :returns: list of cluster label for each data point
    """
    clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=100,
            cluster_selection_epsilon=0.2,
            cluster_selection_method = 'leaf'
    )

    clusterer.fit(X)

    return clusterer.labels_


def cluster(fps):
    """Generic function to trigger the clustering chain.

    :param fps: molecule fingerprints
    :returns: list of cluster label for each data point
    """

    X = create_UMAP(fps)
    labels = cluster_hdbscan(X)

    return labels


def extract_clusters(df, labels):
    """Annotate each row in a dataframe with a cluster label.

    Add a new column with the cluster number the row belongs to.

    :param df: Pandas dataframe
    :param labels: list with cluster labels
    :returns: annotated Pandas dataframe
    """

    df['cluster'] = -99

    for i in set(labels):
        idx = np.where(labels == i)
        df['cluster'].iloc[idx] = i

    return df


def subsample_cluster(df, labels, N, response_label):
    """Sub-sample each cluster by choosing the top N (by dG) compounds.

    :param df: Pandas dataframe
    :param labels: list with cluster labels
    :param N: number of compounds to choose from each cluster
    :returns: new Pandas dataframe with top N compounds
    """

    cluster_df = pd.DataFrame()

    for i in set(labels):
        subset = df[df['cluster'] == i]
        subsample = get_topN(subset, N, response_label)
        cluster_df = pd.concat([cluster_df, subsample])

    return cluster_df


def get_topN(df, N, response_label):
    """Return the top N (by dG) of a Pandas data frame,

    :param df: Pandas dataframe
    :param N: number of compounds to choose from each cluster
    """

    return df.sort_values(by=[response_label]).head(N)


def main():
    args = parse_command_line()

    smiles_column = args.input_column
    response_column = args.response_column

    out_csv = os.path.abspath(args.output_csv)

    CL_results = []

    for filename in args.csv_filenames:
        df = pd.read_csv(filename)
        CL_results.append(df)
        logging.info(f"Read {filename}, {len(df)} molecules")

    all_results = pd.concat(CL_results)

    good_results = filter_df(all_results, response_column)
    logging.info(f"Filtered out {len(good_results)} good molecules")

    good_results = good_results.drop_duplicates(subset=[smiles_column], keep=False)
    logging.info(f"Using {len(good_results)} good molecules after deduplication")

    logging.info(f"dG_min = {min(good_results[response_column]):.2f}")

    if not args.no_clustering:
        logging.info("Creating RDKit molecules")
        good_results.loc[:, 'Mol'] = \
                good_results[smiles_column].apply(Chem.MolFromSmiles)

        logging.info("Computing RDKit fingerprints")
        good_results.loc[:, 'FP'] = good_results['Mol'].apply(Chem.RDKFingerprint)

        logging.info("Clustering with HDBSCAN on UMAP")
        labels = cluster(list(good_results['FP']))

        n_clusters = len(set(labels))
        compounds_per_cluster = 1 + args.batch_size // n_clusters

        logging.info(f"Found {n_clusters} clusters (including outlier cluster)")
        logging.info(f"Extracting {compounds_per_cluster} compounds from each cluster")

        results = extract_clusters(good_results, labels)
        results = subsample_cluster(results, labels, compounds_per_cluster,
                                    response_column)

        results = results.drop(['Mol', 'FP'], axis=1)

        logging.info("Dropping molecules with highest dG to fit requested batch size")
        results = results.sort_values(by=[response_column])
        results = results[:args.batch_size]
    else:
        logging.info(f"Retrieving top {args.batch_size} compounds")
        results = get_topN(good_results, args.batch_size, response_column)

    logging.info(f"Writing CSV with {len(results)} compounds to {out_csv}")
    results.to_csv(out_csv, index=False)


if __name__ == '__main__':
    main()

    logging.shutdown()
