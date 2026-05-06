"""
Mapper module for the Genie pipeline.

Applies inverse folding to generate candidate sequences from backbone
structures, then folds those sequences back into 3D structures using
a configurable folding model.
"""

import os
import glob
import json
import time
import logging
import re
from tqdm import tqdm

from genie3.evaluation.model.fold.registry import get_fold_model
from genie3.evaluation.model.inverse_fold.registry import get_inverse_fold_model


AF_PREDICTION_FILE_RE = re.compile(
    r"^(?:unrelaxed|relaxed)_rank_\d{3}_alphafold2_(?:ptm|multimer_v3)_model_\d_seed_\d{3}\.(?:pdb|cif)$"
)


class Mapper():
    """
    Applies inverse folding and structure prediction to a set of PDB files.

    Initializes the inverse fold and fold models once, then processes each
    PDB file by generating sequences and predicting their 3D structures.
    """

    def __init__(
        self,
        version,
        inverse_fold_mode,
        inverse_fold_model_name,
        inverse_fold_num_seq,
        fold_mode,
        fold_model_name,
        fold_backend,
        fold_num_models,
        fold_num_recycles,
        verbose,
        device,
        datadir,
    ):
        """
        Initialize the Mapper with inverse fold and fold models.

        Args:
            version: Pipeline version string (e.g. 'binder', 'scaffold')
            inverse_fold_mode: Mode for the inverse folding model
            inverse_fold_model_name: Name of the inverse folding model
            inverse_fold_num_seq: Number of sequences to sample per structure
            fold_mode: Mode for the folding model
            fold_model_name: Name of the folding model
            device: GPU device index to use
            datadir: Path to the data directory
        """
        self.version = version
        self.inverse_fold_mode = inverse_fold_mode
        self.inverse_fold_model_name = inverse_fold_model_name
        self.fold_mode = fold_mode
        self.fold_model_name = fold_model_name
        self.fold_backend = fold_backend
        self.fold_num_models = fold_num_models
        self.fold_num_recycles = fold_num_recycles
        self.verbose = verbose
        self.device = device
        self.datadir = datadir

        logging.info(
            f'[{self.device}] Initializing inverse-fold model '
            f'(name={self.inverse_fold_model_name}, mode={self.inverse_fold_mode})'
        )
        init_start_time = time.perf_counter()
        self.inverse_fold_model = get_inverse_fold_model(
            name=self.inverse_fold_model_name,
            mode=self.inverse_fold_mode,
            version=self.version,
            device=self.device,
            datadir=self.datadir,
            num_samples=inverse_fold_num_seq,
        )
        self.inverse_fold_model_init_seconds = time.perf_counter() - init_start_time
        logging.info(
            f'[{self.device}] Inverse-fold model ready '
            f'(seconds={self.inverse_fold_model_init_seconds:.3f})'
        )

        logging.info(
            f'[{self.device}] Initializing fold model '
            f'(name={self.fold_model_name}, mode={self.fold_mode}, backend={self.fold_backend})'
        )
        init_start_time = time.perf_counter()
        self.fold_model = get_fold_model(
            name=self.fold_model_name,
            mode=self.fold_mode,
            version=self.version,
            device=self.device,
            datadir=self.datadir,
            preload=(
                (self.fold_model_name == 'colabfold' and self.fold_backend == 'streamlined') or
                self.fold_model_name == 'esmfold'
            ),
            backend=self.fold_backend,
            num_models=self.fold_num_models,
            num_recycles=self.fold_num_recycles,
            verbose=self.verbose,
        )
        self.fold_model_init_seconds = time.perf_counter() - init_start_time
        logging.info(
            f'[{self.device}] Fold model ready '
            f'(seconds={self.fold_model_init_seconds:.3f})'
        )

    def map(self, rootdir: str, verbose: bool):
        """
        Process all PDB files in rootdir through inverse folding and structure prediction.

        Runs inverse folding on all PDB files in rootdir/pdbs to generate sequences,
        then folds each sequence and saves the resulting structures.

        Args:
            rootdir: Root directory containing a 'pdbs' subdirectory
            verbose: Whether to display progress bars
        """

        ##################
        ###   Set up   ###
        ##################

        # Validate input directory
        pdbs_dir = os.path.join(rootdir, 'pdbs')
        if not os.path.exists(pdbs_dir):
            logging.error(f'Missing directory for {pdbs_dir}')
            exit(0)
        pdb_filepaths = sorted(glob.glob(os.path.join(pdbs_dir, '*.pdb')))
        n_backbones = len(pdb_filepaths)
        logging.info(
            f'[{self.device}] Inverse folding started '
            f'(model={self.inverse_fold_model_name}, backbones={n_backbones})'
        )

        ########################
        ###   Inverse fold   ###
        ########################

        # Create output directory
        sequences_dir = os.path.join(rootdir, 'sequences')
        if os.path.exists(sequences_dir):
            logging.error(f'Directory existed: {sequences_dir}')
            exit(0)
        os.makedirs(sequences_dir)

        runtime_profile = {
            'generated_structures': len(glob.glob(os.path.join(pdbs_dir, '*.pdb'))),
            'inverse_fold_model_init_seconds': self.inverse_fold_model_init_seconds,
            'fold_model_init_seconds': self.fold_model_init_seconds,
            'fold_model_name': self.fold_model_name,
            'fold_mode': self.fold_mode,
            'fold_num_models': getattr(self.fold_model, 'num_models', None),
            'fold_num_recycles': getattr(self.fold_model, 'num_recycles', None),
        }

        start_time = time.perf_counter()
        self.inverse_fold_model.inverse_fold(
            pdbs_dir=pdbs_dir,
            sequences_dir=sequences_dir
        )
        runtime_profile['inverse_fold_seconds'] = time.perf_counter() - start_time
        sequence_filepaths = sorted(glob.glob(os.path.join(sequences_dir, '*.fasta')))
        logging.info(
            f'[{self.device}] Inverse folding finished '
            f'(sequences={len(sequence_filepaths)}, '
            f'seconds={runtime_profile["inverse_fold_seconds"]:.3f})'
        )

        ################
        ###   Fold   ###
        ################

        # Create output directory
        structures_dir = os.path.join(rootdir, 'structures')
        if os.path.exists(structures_dir):
            logging.error(f'Directory existed: {structures_dir}')
            exit(0)
        os.makedirs(structures_dir)

        # Pack fasta files
        start_time = time.perf_counter()
        lines = []
        sample_names = []
        for sequence_filepath in tqdm(
            glob.glob(os.path.join(sequences_dir, '*.fasta')),
            desc=f'[{self.device}] Preparing', disable=not verbose
        ):
            domain_name = sequence_filepath.split('/')[-1].split('.')[0]
            with open(sequence_filepath) as file:
                resample_index = 0
                for line in file:
                    if line.startswith('>'):
                        sample_name = f'{domain_name}-resample_{resample_index}'
                        lines.append(f'>{sample_name}\n')
                        sample_names.append(sample_name)
                        resample_index += 1
                    else:
                        lines.append(line)
        input_fasta_filepath = os.path.join(structures_dir, 'input.fasta')
        with open(input_fasta_filepath, 'w') as file:
            file.write(''.join(lines))
        runtime_profile['generated_sequences'] = len(sample_names)
        runtime_profile['pack_input_fasta_seconds'] = time.perf_counter() - start_time
        logging.info(
            f'[{self.device}] Sequence packing finished '
            f'(generated_sequences={runtime_profile["generated_sequences"]}, '
            f'seconds={runtime_profile["pack_input_fasta_seconds"]:.3f})'
        )

        # Run
        logging.info(
            f'[{self.device}] Structure prediction started '
            f'(model={self.fold_model_name}, mode={self.fold_mode}, '
            f'sequences={runtime_profile["generated_sequences"]})'
        )
        start_time = time.perf_counter()
        self.fold_model.fold(structures_dir)
        runtime_profile['fold_seconds'] = time.perf_counter() - start_time
        runtime_profile['predicted_structures'] = self._count_prediction_structures(structures_dir)
        logging.info(
            f'[{self.device}] Structure prediction finished '
            f'(predicted_structures={runtime_profile["predicted_structures"]}, '
            f'seconds={runtime_profile["fold_seconds"]:.3f})'
        )

        logging.info(
            f'[{self.device}] runtime profile '
            + ' '.join([
                f'{key}={value:.3f}s'
                for key, value in runtime_profile.items()
                if isinstance(value, (int, float))
            ])
        )
        with open(os.path.join(rootdir, 'evaluation_map_stats.json'), 'w') as file:
            json.dump(runtime_profile, file, indent=2, sort_keys=True)
            file.write('\n')

    def _count_prediction_structures(self, structures_dir: str) -> int:
        count = 0
        for filepath in glob.glob(os.path.join(structures_dir, '*', '*')):
            if not os.path.isfile(filepath):
                continue
            basename = os.path.basename(filepath)
            parent = os.path.basename(os.path.dirname(filepath))
            if AF_PREDICTION_FILE_RE.match(basename) or basename == f'{parent}.pdb':
                count += 1
        return count
