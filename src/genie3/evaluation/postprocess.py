"""
Postprocessing stage for multi-node distributed evaluation.

Transfers results from node-specific directories back to the root,
aggregates them, and runs the final reduction step.
"""

import os
import sys
import glob
import shutil 
import logging
import argparse

from genie3.evaluation.pipeline import Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_postprocess_stage(params):
    """
    Transfer and aggregate per-node results into the root directory.

    Copies 'sequences' and 'structures' outputs from each node directory
    into the root-level equivalents.

    Args:
        params: Dictionary with keys:
            rootdir (str): Root directory
            n_node (int): Number of nodes
    """
    
    # Transfer
    for dirpath in glob.glob(os.path.join(params['rootdir'], 'nodes', 'node_*', '')):

        # Transfer files in pdbs directory
        for filepath in glob.glob(os.path.join(dirpath, 'pdbs', '*.pdb')):
            filename = filepath.split('/')[-1]
            shutil.copyfile(
                filepath,
                os.path.join(params['rootdir'], 'pdbs', filename)
            )
        
        # Transfer files in sequences directory
        for filepath in glob.glob(os.path.join(dirpath, 'sequences', '*.fasta')):
            filename = filepath.split('/')[-1]
            shutil.copyfile(
                filepath,
                os.path.join(params['rootdir'], 'sequences', filename)
            )
        
        # Transfer files in structures directory
        for subdirpath in glob.glob(os.path.join(dirpath, 'structures', '*', '')):
            dirname = subdirpath.split('/')[-2]
            shutil.copytree(
                subdirpath,
                os.path.join(params['rootdir'], 'structures', dirname)
            )
    
    # Aggregate input.fasta file in structure directory
    input_fasta_lines = []
    for dirpath in glob.glob(os.path.join(params['rootdir'], 'nodes', 'node_*', '')):
        with open(os.path.join(dirpath, 'structures', 'input.fasta')) as file:
            input_fasta_lines.extend(file.readlines())
    with open(os.path.join(params['rootdir'], 'structures', 'input.fasta'), 'w') as file:
        file.write(''.join(input_fasta_lines))
    logging.info('Transfer completed!')
    
    # Define runner
    runner = Runner()

    # Aggregate results
    runner.reduce(
        version=params['version'],
        rootdir=params['rootdir'],
        verbose=params['verbose'],
        params=params
    )

def main(args):
    """
    Entry point for the postprocessing stage.

    Validates the root directory, checks for prior completion or existing
    node directories, then runs the postprocessing stage and reduction.

    Args:
        args: Parsed command-line arguments with fields:
            rootdir (str): Root directory
            n_node (int): Number of nodes
            version (str): Pipeline version string
            verbose (bool): Whether to log progress
            datadir (str): Data directory path
            calc_novelty (bool): Whether to compute novelty metrics
            fold_mode (str): Folding mode
            fold_model_name (str): Name of the fold model
    """

    # Check for completion
    if os.path.exists(os.path.join(args.rootdir, 'results', 'eval.done')):
        logging.info('Evaluation already completed!')
        exit(0)
    
    # Check evaluation status
    n_task, n_complete_task = 0, 0
    for dirpath in glob.glob(os.path.join(args.rootdir, 'nodes', 'node_*', '')):
        n_task += 1
        if os.path.exists(os.path.join(dirpath, 'results', 'eval.done')):
            n_complete_task += 1
    if n_task == 0:
        logging.error('No distributed task registered!')
        exit(0)
    if n_task != n_complete_task:
        logging.info(f'Status: {n_complete_task} / {n_task}')
        logging.info('Waiting for evaluations to complete')
        exit(0)
    else:
        logging.info('Evaluation completed!')

    # Set up directory
    dirnames = ['pdbs', 'sequences', 'structures', 'results']
    for dirname in dirnames:
        if os.path.exists(os.path.join(args.rootdir, dirname)):
            shutil.rmtree(os.path.join(args.rootdir, dirname))
        if dirname != 'results':
            os.makedirs(os.path.join(args.rootdir, dirname))
    
    # Define parameters
    params = vars(args)

    # Run
    run_postprocess_stage(params)

    # Mark success
    open(os.path.join(args.rootdir, 'results', 'eval.done'), 'w').close()
    logging.info('Finished postprocessing!')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
    parser.add_argument('--version', type=str, help='Pipeline version', required=True)
    parser.add_argument('--verbose', help='Whether to log evaluation process', action='store_true', default=False)
    parser.add_argument('--datadir', type=str, help='Data directory', default=None)
    parser.add_argument('--calc_novelty', help='Whether to calculate novelty', action='store_true', default=False)

    parser.add_argument('--fold_mode', type=str, help='Folding mode', default='msa')
    parser.add_argument('--fold_model_name', type=str, help='Fold model name', default='colabfold')
    args = parser.parse_args()
    
    # Run
    main(args)
