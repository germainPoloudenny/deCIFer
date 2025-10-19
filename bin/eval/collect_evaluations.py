#!/usr/bin/env python3

import os
import gzip
import pickle
import argparse
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from multiprocessing import Pool, cpu_count

from bin.eval.show_eval_metrics import show_metrics

from decifer.utility import extract_space_group_symbol, space_group_symbol_to_number

def rwp(sample, gen):
    """
    Calculates the residual (un)weighted profile between a sample and a generated XRD pattern
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))

def process_file(file_path):
    """Processes a single .pkl.gz file."""
    try:
        with gzip.open(file_path, 'rb') as f:
            row = pickle.load(f)

       # If successful generation, count 
        if 'success' not in row['status']:
            return None

        # Extract Validity
        formula_validity = row['validity']['formula']
        bond_length_validity = row['validity']['bond_length']
        spacegroup_validity = row['validity']['spacegroup']
        site_multiplicity_validity = row['validity']['site_multiplicity']
        valid = all([formula_validity, bond_length_validity, spacegroup_validity, site_multiplicity_validity])

        # Extract CIFs and XRD (Sample)
        cif_sample = row['cif_string_sample']
        xrd_q_continuous_sample = row['xrd_clean_sample']['q']
        xrd_iq_continuous_sample = row['xrd_clean_sample']['iq']
        xrd_q_discrete_sample = row['xrd_clean_sample']['q_disc']
        xrd_iq_discrete_sample = row['xrd_clean_sample']['iq_disc']

        # Extract CIFs and XRD (Generated)
        cif_gen = row['cif_string_gen']
        xrd_q_continuous_gen = row['xrd_clean_gen']['q']
        xrd_iq_continuous_gen = row['xrd_clean_gen']['iq']
        xrd_q_discrete_gen = row['xrd_clean_gen']['q_disc']
        xrd_iq_discrete_gen = row['xrd_clean_gen']['iq_disc']
        
        # Normalize for wasserstein
        # Wasserstein Distance
        xrd_iq_discrete_sample_normed = xrd_iq_discrete_sample / np.sum(xrd_iq_discrete_sample)
        xrd_iq_discrete_gen_normed = xrd_iq_discrete_gen / np.sum(xrd_iq_discrete_gen)
        wd_value = wasserstein_distance(xrd_q_discrete_sample, xrd_q_discrete_gen, u_weights=xrd_iq_discrete_sample_normed, v_weights=xrd_iq_discrete_gen_normed)

        # Rwp
        rwp_value = rwp(xrd_iq_continuous_sample, xrd_iq_continuous_gen)

        # L2 distance between continuous diffractograms
        l2_distance = float(np.linalg.norm(
            np.asarray(xrd_iq_continuous_sample) - np.asarray(xrd_iq_continuous_gen)
        ))

        # RMSD
        rmsd_value = row['rmsd']

        # Sequence lengths
        seq_len_sample = row['seq_len_sample']
        seq_len_gen = row['seq_len_gen']

        # Extract space group
        spacegroup_sym_sample = extract_space_group_symbol(cif_sample)
        spacegroup_num_sample = space_group_symbol_to_number(spacegroup_sym_sample)
        spacegroup_num_sample = int(spacegroup_num_sample) if spacegroup_num_sample is not None else np.nan

        spacegroup_sym_gen = extract_space_group_symbol(cif_gen)
        spacegroup_num_gen = space_group_symbol_to_number(spacegroup_sym_gen)
        spacegroup_num_gen = int(spacegroup_num_gen) if spacegroup_num_gen is not None else np.nan

        out_dict = {
            'index': row.get('index'),
            'cif_name': row.get('cif_name'),
            'rep': row.get('rep'),
            'dataset_name': row.get('dataset_name'),
            'model_name': row.get('model_name'),
            'rwp': rwp_value,
            'l2_distance': l2_distance,
            'wd': wd_value,
            'rmsd': rmsd_value,
            'cif_sample': cif_sample,
            'xrd_q_discrete_sample': xrd_q_discrete_sample,
            'xrd_iq_discrete_sample': xrd_iq_discrete_sample,
            'xrd_q_continuous_sample': xrd_q_continuous_sample,
            'xrd_iq_continuous_sample': xrd_iq_continuous_sample,
            'spacegroup_sym_sample': spacegroup_sym_sample,
            'spacegroup_num_sample': spacegroup_num_sample,
            'seq_len_sample': seq_len_sample,
            'cif_gen': cif_gen,
            'xrd_q_discrete_gen': xrd_q_discrete_gen,
            'xrd_iq_discrete_gen': xrd_iq_discrete_gen,
            'xrd_q_continuous_gen': xrd_q_continuous_gen,
            'xrd_iq_continuous_gen': xrd_iq_continuous_gen,
            'seq_len_gen': seq_len_gen,
            'spacegroup_sym_gen': spacegroup_sym_gen,
            'spacegroup_num_gen': spacegroup_num_gen,
            'formula_validity': formula_validity,
            'spacegroup_validity': spacegroup_validity,
            'bond_length_validity': bond_length_validity,
            'site_multiplicity_validity': site_multiplicity_validity,
            'validity': valid,
        }
        return out_dict
    except Exception as e:
        raise e
        print(f"Error processing file {file_path}: {e}")
        return None

def process(folder, debug_max=None, top_k=None, top_k_metric: str = "rwp") -> pd.DataFrame:
    """Processes all files in the given folder using multiprocessing."""
    # Get list of files
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl.gz')]
    if debug_max is not None:
        files = files[:debug_max]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing files..."))

    # Filter out None results and convert to DataFrame
    data_list = [res for res in results if res is not None]
    df = pd.DataFrame(data_list)

    if top_k is not None and top_k > 0 and not df.empty:
        metric_column = top_k_metric
        if metric_column == "l2":
            metric_column = "l2_distance"
        if metric_column not in df.columns:
            raise ValueError(f"Metric '{top_k_metric}' not available for sorting.")
        if 'index' in df.columns:
            group_column = 'index'
        elif 'cif_name' in df.columns:
            group_column = 'cif_name'
        else:
            raise ValueError(
                "Cannot apply per-sample top-k filtering because neither 'index' nor 'cif_name' is present in the data."
            )

        sort_columns = [group_column, metric_column]
        if 'rep' in df.columns:
            sort_columns.append('rep')
        df = (
            df.sort_values(by=sort_columns, ascending=True)
              .groupby(group_column, group_keys=False)
              .head(top_k)
        )

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-folder-paths", nargs='+', required=True, help="Provide a list of folder paths")
    parser.add_argument("--output-folder", type=str, default='.')
    parser.add_argument("--debug_max", type=int, default=0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "Keep only the top-K rows per sample with the lowest selected metric "
            "(0 disables the filter)"
        ),
    )
    parser.add_argument(
        "--top-k-metric",
        choices=["rwp", "l2"],
        default="rwp",
        help="Metric used to rank rows when applying --top-k.",
    )
    args = parser.parse_args()
    if args.debug_max == 0:
        args.debug_max = None

    top_k = args.top_k if args.top_k > 0 else None
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Loop over folders
    folder_names = [path.split("/")[-1] for path in args.eval_folder_paths]
    for label, path in zip(folder_names, args.eval_folder_paths):
        df = process(path, args.debug_max, top_k, args.top_k_metric)
        pickle_path = os.path.join(args.output_folder, label + '.pkl.gz')
        df.to_pickle(pickle_path)
        # ``collect_evaluations`` n'applique pas de seuil sur le RMSD : les
        # valeurs sont exportées telles quelles et ``show_metrics`` se contente
        # de les résumer. Si un taux de "match" doit être calculé, il faut donc
        # fournir un seuil explicite dans l'étape d'analyse suivante.
        show_metrics([pickle_path])
