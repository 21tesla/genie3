"""
Preprocessing stage for multi-node distributed evaluation.

Distributes PDB files across node-specific subdirectories so that each
node can independently run the evaluation pipeline in parallel.
"""

import os
import sys
import glob
import math
import shutil 
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def run_preprocess_stage(params):
    """
    Distribute PDB files across node subdirectories.

    Splits PDB files from rootdir/pdbs evenly across n_node node directories,
    creating rootdir/nodes/node_<i>/pdbs/ for each node.

    Args:
        params: Dictionary with keys:
            rootdir (str): Root directory containing the 'pdbs' subdirectory
            n_node (int): Number of nodes to distribute across
    """

    # Allocate tasks to nodes
    n_node = params['n_node']
    tasks = glob.glob(os.path.join(params['rootdir'], 'pdbs', '*.pdb'))
    tasks = sorted(tasks, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    n_task_per_node = math.ceil(len(tasks) / n_node)
    for node_id in range(n_node):

        # Set up node directory
        basedir = os.path.join(params['rootdir'], 'nodes', f'node_{node_id}', 'pdbs')
        os.makedirs(basedir)

        # Transfer pdb files
        start_task_index = node_id * n_task_per_node
        end_task_index = (node_id + 1) * n_task_per_node
        for filepath in tasks[start_task_index:end_task_index]:
            filename = filepath.split('/')[-1]
            shutil.copyfile(
                filepath,
                os.path.join(basedir, filename)
            )

def main(args):
    """
    Entry point for the preprocessing stage.

    Validates the root directory, checks for prior completion or existing
    node directories, then runs the preprocessing stage.

    Args:
        args: Parsed command-line arguments with fields:
            rootdir (str): Root directory
            n_node (int): Number of nodes
            clean (bool): Whether to remove existing node directories
    """

    # Sanity check
    if not os.path.exists(os.path.join(args.rootdir, 'pdbs')):
        logging.error('Preprocessing requires the root directory to contain pdbs.')
        logging.error('The root directory need to be problem-specific for setting up multi-node processing.')
        exit(0)

    # Check for completion
    if os.path.exists(os.path.join(args.rootdir, 'results', 'eval.done')):
        logging.info('Evaluation already completed!')
        exit(0)
    elif os.path.exists(os.path.join(args.rootdir, 'nodes')):
        if args.clean:
            shutil.rmtree(os.path.join(args.rootdir, 'nodes'))
        else:
            logging.info('Already preprocessed!')
            exit(0)

    # Define parameters
    params = vars(args)

    # Run
    run_preprocess_stage(params)

    # Mark success
    logging.info('Finished preprocessing!')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, help='Root directory', required=True)
    parser.add_argument('--n_node', type=int, help='Number of nodes to distribute', default=1)
    parser.add_argument('--clean', help='Whether to clean up previous intermediate evaluations', action='store_true', default=False)
    args = parser.parse_args()
    
    # Run
    main(args)