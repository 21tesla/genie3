"""
Abstract base class for pipeline reducers.

Defines the common interface for compiling, filtering, and scoring
designed structures, as well as shared utilities for diversity and
novelty computation via FoldSeek.
"""

import os
import glob
import shutil
import logging
import subprocess
from contextlib import nullcontext
from abc import ABC, abstractmethod

from genie3.evaluation.model.fold.registry import get_fold_model


class Reducer(ABC):
    """Abstract base class for pipeline result reducers."""

    def __init__(self, version, datadir, fold_model_name, fold_mode, calc_novelty):
        """
        Initialize the Reducer.

        Args:
            version: Pipeline version string (e.g. 'binder', 'scaffold')
            datadir: Path to the data directory
            fold_model_name: Name of the fold model used for compilation
            fold_mode: Mode for the fold model
            calc_novelty: Whether to compute novelty metrics against databases
        """
        self.version = version
        self.datadir = datadir
        self.fold_model_name = fold_model_name
        self.fold_mode = fold_mode
        self.calc_novelty = calc_novelty

        self.fold_model = get_fold_model(
            name=self.fold_model_name,
            mode=self.fold_mode,
            version=self.version,
            datadir=self.datadir,
            preload=False
        )
        self._subprocess_verbose = False

    @abstractmethod
    def _validate(self):
        """Validate that the reducer configuration is valid for this version."""
        raise NotImplemented

    @abstractmethod
    def _compile(self, pdbs_dir: str, structures_dir: str, results_dir: str, verbose: bool):
        """
        Compile per-design metrics from folded structures into a results CSV.

        Args:
            pdbs_dir: Directory containing original generated PDB files
            structures_dir: Directory containing folded structure outputs
            results_dir: Directory to write compiled results
            verbose: Whether to display a progress bar
        """
        raise NotImplemented
    
    @abstractmethod
    def _filter(self, results_dir: str):
        """
        Filter compiled results and copy successful designs to output directories.

        Args:
            results_dir: Directory containing the compiled info.csv
        """
        raise NotImplemented
    
    def _add_successes_dir(self, successes_dir: str):
        """
        Register a directory of successful designs for diversity/novelty analysis.

        Args:
            successes_dir: Path to directory containing successful PDB files
        """
        if not hasattr(self, 'list_successes_dir'):
            self.list_successes_dir = []
        self.list_successes_dir.append(successes_dir)

    def _compute_diversity(self, successes_dir: str):
        """
        Cluster successful designs at multiple TM-score thresholds and save results.

        Runs FoldSeek clustering at TM-score thresholds of 0.5, 0.6, and 0.8,
        then writes a CSV mapping each design name to its cluster ID at each
        threshold.

        Args:
            successes_dir: Directory containing successful PDB/CIF files
        """

        ##################
        ###   Set up   ###
        ##################

        # Parse
        basedir = successes_dir.rsplit('/', 1)[0]
        name = successes_dir.rsplit('/', 1)[1]

        # Create output filepath
        output_cluster_filepath = os.path.join(basedir, f'{name}_cluster.csv')
        if os.path.exists(output_cluster_filepath):
            logging.error(f'Output filepath existed: {output_cluster_filepath}')
            exit(0)

        ###################
        ###   Cluster   ###
        ###################

        tm_0_5_name_to_cluster_id = self._cluster(
            successes_dir=successes_dir,
            tmscore_threshold=0.5
        )
        tm_0_6_name_to_cluster_id = self._cluster(
            successes_dir=successes_dir,
            tmscore_threshold=0.6
        )
        tm_0_8_name_to_cluster_id = self._cluster(
            successes_dir=successes_dir,
            tmscore_threshold=0.8
        )

        #######################
        ###   Postprocess   ###
        #######################

        # Sanity check
        assert len(tm_0_5_name_to_cluster_id) == len(tm_0_6_name_to_cluster_id)
        assert len(tm_0_5_name_to_cluster_id) == len(tm_0_8_name_to_cluster_id)

        # Save
        with open(output_cluster_filepath, 'w') as file:
            columns = ['name', 'domain', 'tm_0_5_cluster_id', 'tm_0_6_cluster_id', 'tm_0_8_cluster_id']
            file.write(','.join(columns) + '\n')
            for name in tm_0_5_name_to_cluster_id:
                domain = name.split('-')[0]
                file.write(','.join([
                    name,
                    domain,
                    tm_0_5_name_to_cluster_id[name],
                    tm_0_6_name_to_cluster_id[name],
                    tm_0_8_name_to_cluster_id[name]
                ]) + '\n')

    def _cluster(self, successes_dir: str, tmscore_threshold: float):
        """
        Cluster structures using FoldSeek at a given TM-score threshold.

        Args:
            successes_dir: Directory containing PDB/CIF files to cluster
            tmscore_threshold: TM-score cutoff for cluster membership

        Returns:
            dict: Mapping from structure name to cluster representative name
        """

        ##################
        ###   Set up   ###
        ##################

        # Parse
        basedir = successes_dir.rsplit('/', 1)[0]

        # Check for empty input directory
        if len(
            list(glob.glob(os.path.join(successes_dir, '*.pdb'))) +
            list(glob.glob(os.path.join(successes_dir, '*.cif')))
        ) == 0:
            return {}

        # Create output directory
        diversity_dir = os.path.join(basedir, f'diversity')
        if os.path.exists(diversity_dir):
            logging.error(f'Output directory existed: {diversity_dir}')
            exit(0)
        os.makedirs(diversity_dir)

        # Create temporary directory
        temp_dir = os.path.join(basedir, f'temp')
        if os.path.exists(temp_dir):
            logging.error(f'Temporary directory existed: {temp_dir}')
            exit(0)
        os.makedirs(temp_dir)

        ###############################
        ###   Compute pairwise TM   ###
        ###############################

        # Run FoldSeek clustering
        cmd = ' '.join([
            'foldseek easy-cluster',
            successes_dir,
            os.path.join(diversity_dir, 'diversity'),
            temp_dir,
            '--cov-mode 0',
            '--alignment-type 1',
            '--min-seq-id 0',
            f'--tmscore-threshold {tmscore_threshold}'
        ])
        self._run_command(cmd)

        # Generate mapping
        name_to_cluster_id = {}
        with open(os.path.join(diversity_dir, 'diversity_cluster.tsv')) as file:
            for line in file:
                elts = line.strip().split()
                cluster_id, name = elts[0], elts[1]
                name_to_cluster_id[name] = cluster_id
        
        ####################
        ###   Clean up   ###
        ####################

        shutil.rmtree(diversity_dir)
        shutil.rmtree(temp_dir)

        return name_to_cluster_id

    def _compute_novelty(self, successes_dir: str):
        """
        Search successful designs against structural databases and save novelty metrics.

        Searches against the PDB and afdbreps_256 databases using FoldSeek,
        then writes a CSV with the closest known structure and its TM-score
        and lDDT for each design.

        Args:
            successes_dir: Directory containing successful PDB files
        """

        ##################
        ###   Set up   ###
        ##################

        # Parse
        basedir = successes_dir.rsplit('/', 1)[0]
        name = successes_dir.rsplit('/', 1)[1]

        # Create output filepath
        output_novelty_filepath = os.path.join(basedir, f'{name}_novelty.csv')
        if os.path.exists(output_novelty_filepath):
            logging.error(f'Output filepath existed: {output_novelty_filepath}')
            exit(0)
        
        ##################
        ###   Search   ###
        ##################

        pdb_name_to_closest_structure = self._search(
            successes_dir=successes_dir,
            database='pdb'
        )

        afdbreps_256_name_to_closest_structure = self._search(
            successes_dir=successes_dir,
            database='afdbreps_256'
        )

        #######################
        ###   Postprocess   ###
        #######################

        # Parse names
        names = [
            filepath.split('/')[-1].split('.')[0]
            for filepath in glob.glob(os.path.join(successes_dir, '*.pdb'))
        ]

        # Save
        with open(output_novelty_filepath, 'w') as file:
            columns = [
                'name', 'domain',
                'pdb_closest_structure', 'pdb_tm', 'pdb_lddt', 'pdb_has_aln',
                'afdbreps_256_closest_structure', 'afdbreps_256_tm', 'afdbreps_256_lddt', 'afdbreps_256_has_aln'
            ]
            file.write(','.join(columns) + '\n')
            for name in names:
                domain = name.split('-')[0]
                elts = [name, domain]
                if name in pdb_name_to_closest_structure:
                    elts.extend([
                        pdb_name_to_closest_structure[name][0],
                        str(pdb_name_to_closest_structure[name][1]),
                        str(pdb_name_to_closest_structure[name][2]),
                        'True'
                    ])
                else:
                    elts.extend([
                        '',      # No closest structure
                        '0',     # Default tm score to 0
                        '0',     # Default lddt to 0
                        'False'  # No alignment
                    ])
                if name in afdbreps_256_name_to_closest_structure:
                    elts.extend([
                        afdbreps_256_name_to_closest_structure[name][0],
                        str(afdbreps_256_name_to_closest_structure[name][1]),
                        str(afdbreps_256_name_to_closest_structure[name][2]),
                        'True'
                    ])
                else:
                    elts.extend([
                        '',      # No closest structure
                        '0',     # Default tm score to 0
                        '0',     # Default lddt to 0
                        'False'  # No alignment
                    ])
                file.write(','.join(elts) + '\n')

    def _search(self, successes_dir: str, database: str):
        """
        Search structures against a FoldSeek database for novelty assessment.

        Args:
            successes_dir: Directory containing PDB files to search
            database: Database name (e.g. 'pdb', 'afdbreps_256')

        Returns:
            dict: Mapping from structure name to (closest_name, tm_score, lddt)
        """

        ##################
        ###   Set up   ###
        ##################

        # Parse
        basedir = successes_dir.rsplit('/', 1)[0]

        # Check for empty input directory
        if len(list(glob.glob(os.path.join(successes_dir, '*.pdb')))) == 0:
            return {}

        # Create output directory
        novelty_dir = os.path.join(basedir, f'novelty')
        if os.path.exists(novelty_dir):
            logging.error(f'Output directory existed: {novelty_dir}')
            exit(0)
        os.makedirs(novelty_dir)

        ####################################
        ###   Search closest structure   ###
        ####################################

        # Get database directory
        database_dir = f'databases/{database}'
        if not os.path.exists(database_dir):
            logging.error(f'Missing database: {database}')
            return {}

        # Run FoldSeek searching
        cmd = ' '.join([
            'foldseek easy-search',
            successes_dir,
            os.path.join(database_dir, database),
            os.path.join(novelty_dir, 'out.tsv'),
            os.path.join(novelty_dir, 'temp'),
            '--alignment-type 1',
            '--exhaustive-search',
            '--tmscore-threshold 0.0',
            '--max-seqs 10000000000',
            '--format-output query,target,qtmscore,lddt' # Use qtmscore instead of alntmscore
        ])
        self._run_command(cmd)

        # Generate mapping
        name_to_closest_structure = {}
        with open(os.path.join(novelty_dir, 'out.tsv')) as file:
            for line in file:
                elts = line.strip().split()
                name, novelty_name, novelty_tm, novelty_lddt = elts[0], elts[1], float(elts[2]), float(elts[3])
                if (
                    name not in name_to_closest_structure or
                    novelty_tm > name_to_closest_structure[name][1]
                ):
                    name_to_closest_structure[name] = (novelty_name, novelty_tm, novelty_lddt)
        
        ####################
        ###   Clean up   ###
        ####################

        shutil.rmtree(novelty_dir)

        return name_to_closest_structure

    def reduce(self, rootdir: str, verbose: bool):
        """
        Run the full reduction pipeline: compile, filter, and optionally score.

        Validates input directories, calls _compile to build results CSV,
        calls _filter to extract successful designs, then computes diversity
        and optionally novelty for each registered successes directory.

        Args:
            rootdir: Root directory containing 'pdbs' and 'structures' subdirectories
            verbose: Whether to display progress bars
        """
        
        ##################
        ###   Set up   ###
        ##################

        # Validate input pdb directory
        pdbs_dir = os.path.join(rootdir, 'pdbs')
        if not os.path.exists(pdbs_dir):
            logging.error(f'Missing directory for {pdbs_dir}')
            exit(0)
        
        # Validate input structure directory
        structures_dir = os.path.join(rootdir, 'structures')
        if not os.path.exists(structures_dir):
            logging.error(f'Missing directory for {structures_dir}')
            exit(0)
        self._subprocess_verbose = verbose
        
        ###################
        ###   Compile   ###
        ###################

        # Create output directory
        results_dir = os.path.join(rootdir, 'results')
        if os.path.exists(results_dir):
            logging.error(f'Directory existed: {results_dir}')
            exit(0)
        os.makedirs(results_dir)

        # Compile
        self._compile(
            pdbs_dir=pdbs_dir,
            structures_dir=structures_dir,
            results_dir=results_dir,
            verbose=verbose
        )

        ##################
        ###   Filter   ###
        ##################

        # Validate input info filepath
        info_filepath = os.path.join(results_dir, 'info.csv')
        if not os.path.exists(info_filepath):
            logging.error(f'Missing info: {info_filepath}')
            exit(0)
        
        # Filter
        self._filter(results_dir)

        ###############################
        ###   Diversity / Novelty   ###
        ###############################

        if hasattr(self, 'list_successes_dir'):
            for successes_dir in self.list_successes_dir:
                self._compute_diversity(successes_dir)
                if self.calc_novelty:
                    self._compute_novelty(successes_dir)

    def _run_command(self, cmd: str) -> None:
        """Run an external command, silencing terminal noise unless verbose."""
        logging.debug('[Reducer] Running command: %s', cmd)
        if self._subprocess_verbose:
            result = subprocess.run(cmd, shell=True)
        else:
            log_path = self._resolve_run_log_path()
            stream_context = (
                open(log_path, 'a') if log_path is not None else nullcontext(subprocess.DEVNULL)
            )
            with stream_context as stream:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=stream,
                    stderr=stream,
                )
        if result.returncode != 0:
            raise RuntimeError(f'Command failed with exit code {result.returncode}: {cmd}')

    def _resolve_run_log_path(self) -> str | None:
        """Best-effort lookup of the active Genie3 run log path."""
        for logger_name in ('genie3', ''):
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                base_filename = getattr(handler, 'baseFilename', None)
                if base_filename:
                    return base_filename
        return None
