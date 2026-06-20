"""Create the ChemProp model from the ESMACS data

Reads the SMILES and dG from an ESMACS CSV.  This data is combined with the
aggregated, previous data ("updating" currently means to build all new models
from scratch).  Only the best CV fold from the previous run is used.  After the
model has been built the best fold and its best model within will be written to
stdout as two integers.  All logging goes tostderr and so can be conveniently
redirected.

TODO: Investigate how to update a model and how freezing works.
"""

import os
import sys
import argparse
import logging
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr



import chemprop
from chemprop.train import cross_validate, run_training
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import numpy as np
from rdkit import Chem


__progname__ = "CL-preprocessing"
__version__ = "0.0.0"
__copyright__ = "(C) AstraZeneca 2024"
__authors__ = "Hannes H Loeffler"

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def setup_logging():
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt="[%Y-%m-%d %H:%M:%S %Z]", level=logging.INFO)


@contextmanager
def manage_output(flag):
    """Context manager to redirect stdout and stderr to /dev/null"""

    if not flag:
        yield

    with open(os.devnull, "w") as nowhere:
        with redirect_stderr(nowhere) as err, redirect_stdout(nowhere) as out:
            yield err, out


def parse_command_line():
    parser = argparse.ArgumentParser(
        description=f"{__progname__}: preprocess CSV file(s) from ESMACS "
        "for a REINVENT CL run and create a new, updated ChemProp model",
        epilog=f"v{__version__} {__copyright__}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "csv_filename",
        default=None,
        metavar="FILE",
        type=os.path.abspath,
        help="Input CSV file(s) from a ESMACS run",
    )

    parser.add_argument(
        "-p",
        "--previous-csv-filename",
        metavar="FILE",
        type=os.path.abspath,
        help="CSV file with accumulated data from all previous ESMACS runs"
    )

    parser.add_argument(
        "-c",
        "--chemprop-checkpoint-path",
        metavar="FILE",
        type=os.path.abspath,
        help="Top directory to ChemProp models to load seed model from"
    )

    parser.add_argument(
        "-s",
        "--chemprop-save-dir",
        metavar="FILE",
        type=os.path.abspath,
        help="Save directory for ChemProp models"
    )

    parser.add_argument(
        "-e",
        "--chemprop-epochs",
        metavar="N",
        type=int,
        help="Number of epochs"
    )

    parser.add_argument(
        "-j",
        "--chemprop-hyper-filename",
        metavar="FILE",
        type=os.path.abspath,
        help="ChemProp hyper-parameter JSON file"
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
        default="dG_ESMACS",
        help="Column header for the response in the input CSV file.",
    )

    parser.add_argument(
        "--silent",
        action='store_true',
        help="Suppress logging but still write out final best model"
    )


    return parser.parse_args()


def parse_quiet_log(pathname):
    """Parse quite.log for useful information

    :param pathname: path name to quiet.log
    :returns: model and fold with lowest test RMSE
    """

    folds = []
    models = []
    seeds = []

    with open(pathname, "r") as qfile:
        for line in qfile:
            # we heavily rely on order here...
            if line.startswith("Fold"):
                if models:
                    folds.append(models)

                models = []

            if line.startswith("Model") and "test rmse" in line:
                models.append(float(line[20:]))

            if "Seed" in line:
                seeds.append(float(line[24:]))

    folds.append(models)

    folds = np.array(folds)
    seeds = np.array(seeds)

    best_model = np.unravel_index(folds.argmin(), folds.shape)
    best_fold = seeds.argmin()

    return best_model, best_fold


def make_chemprop_model(checkpoint_path: str, data_path: str, smiles_column: str,
                        response_column: str, save_dir: str, epochs: int = 30,
                        hyper_filename: str = None):
    """Make a new ChemProp model using CV and ensembling

    :param checkpoint_path: directory with ChemProp models to start from
    :param data_path: CSV file from which the data is read
    :param smiles_column: columns in data_path containing the SMILES
    :param response_column: columns in data_path containing the data to train on
    :param save_dir: directory where the new ChemProp models are stored
    :param epochs: maximum number of epochs to run
    :param hyper_filename: JSON file with hyper-parameters, must match models in checkoint_path
    """

    # All arguments must be str
    args = [
        "--data_path", data_path,  # CSV data file
        "--dataset_type", "regression",
        "--split_type", "cv",
        "--split_sizes", "0.8", "0.1", "0.1",
        "--epochs", str(epochs),
        "--num_folds", "5",
        "--aggregation", "norm",
        "--aggregation_norm", "25",
        "--smiles_column", smiles_column,
        "--target_columns", response_column,
        "--save_dir", save_dir,  # output directory
        "--features_generator", "rdkit_2d_normalized", "--no_features_scaling",
        "--save_smiles_split",
    ]

    if os.path.exists(checkpoint_path):
        args.extend(["--checkpoint_dir", checkpoint_path])  # seed model

    # JSON with optimized hyper-parameters
    if hyper_filename and os.path.exists(hyper_filename):
        args.extend(["--config_path", hyper_filename])

    chemprop_args = chemprop.args.TrainArgs().parse_args(args)
    cross_validate(args=chemprop_args, train_func=run_training)


COMBINED_CSV = "_combined.csv"

def main():
    args = parse_command_line()

    if not args.silent:
        setup_logging()

    logging.info(f"Reading aggregated ESMACS data from "
                 f"{args.previous_csv_filename}")
    prev = pd.read_csv(args.previous_csv_filename)

    logging.info(f"Reading new ESMACS data from {args.csv_filename}")
    new = pd.read_csv(args.csv_filename)

    combined = pd.concat((prev, new))

    # Kludge to enforce new column format and order as created by ESMACS
    # because the aggregated ESMACS data may still be in an outdated format
    combined = combined[new.columns]
    combined.to_csv(COMBINED_CSV, index=False)

    quiet_log_filename = os.path.join(args.chemprop_checkpoint_path,
                                       "quiet.log")
    chemprop_checkpoint = ""

    if os.path.exists(quiet_log_filename):
        _, best_fold = parse_quiet_log(quiet_log_filename)
        logging.info(f"Best model in previous model is {best_fold}")
        chemprop_checkpoint = os.path.join(args.chemprop_checkpoint_path,
                                           f"fold_{best_fold}")

    logging.info(f"Running ChemProp {chemprop.__version__}")

    with manage_output(args.silent):
        make_chemprop_model(chemprop_checkpoint, os.path.abspath(COMBINED_CSV),
                            args.input_column, args.response_column,
                            args.chemprop_save_dir, args.chemprop_epochs,
                            args.chemprop_hyper_filename)

    new_quiet_log_filename = os.path.join(args.chemprop_save_dir, "quiet.log")
    best_model, _ = parse_quiet_log(new_quiet_log_filename)
    logging.info(f"Best new model is {best_model}")

    print(best_model[0], best_model[1], flush=True)


if __name__ == '__main__':
    main()

    logging.shutdown()
