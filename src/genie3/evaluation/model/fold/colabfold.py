"""
ColabFold handler for the Genie pipeline.

Wraps the ColabFold tool for protein structure prediction, supporting
single-sequence, MSA, and template-based folding modes.
"""

import os
import re
import copy
import glob
import json
import time
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

from genie3.evaluation.model.fold.base import FoldHandler
from genie3.evaluation.utils.metric import compute_multimer_ca_rmsd, compute_tmscore
from genie3.evaluation.utils.msa import make_complex_msa
from genie3.evaluation.utils.parse import parse_af_filename, parse_fasta, parse_problem_name_from_fasta
from genie3.evaluation.utils.secondary import assign_secondary_structure


class ColabFoldHandler(FoldHandler):
    """FoldHandler implementation using the ColabFold tool."""

    NAME = 'colabfold'

    def __init__(
        self,
        *args,
        backend='subprocess',
        num_models=5,
        num_recycles=20,
        jax_preallocate=None,
        jax_mem_fraction='4.0',
        jax_allocator=None,
        tf_force_unified_memory='1',
        **kwargs,
    ):
        """
        Initialize ColabFoldHandler.

        Args:
            backend: Execution backend ('streamlined' or 'subprocess').
            num_models: Number of AlphaFold models to run (default 5).
            num_recycles: Number of recycling iterations (default None).
            *args / **kwargs: Forwarded to FoldHandler.__init__.
        """
        self.backend = backend
        self.num_models = num_models
        self.num_recycles = num_recycles
        self.jax_preallocate = jax_preallocate
        self.jax_mem_fraction = jax_mem_fraction
        self.jax_allocator = jax_allocator
        self.tf_force_unified_memory = tf_force_unified_memory
        self._model_runner_and_params = None
        self._streamlined = None
        super().__init__(*args, **kwargs)
        if self.preload:
            self._preload()

    def setup(self):
        """
        Validate the version and folding mode for ColabFold.
        """
        if self.version == 'binder':
            if self.datadir is None:
                logging.error('Missing data directory')
                exit(0)
            if self.mode not in ['msa', 'template', 'singleseq']:
                logging.error(f'Invalid colabfold mode for binder design: {self.mode}')
                exit(0)
        else:
            logging.error(f'Invalid version: {self.version}')
            exit(0)

    def _run_subprocess_command(self, cmd: str) -> None:
        """Run an external fold command, silencing terminal noise unless verbose."""
        if self.verbose:
            result = subprocess.run(cmd, shell=True)
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.stdout:
                logging.debug(result.stdout.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                f'[ColabFoldHandler] Command failed with exit code {result.returncode}: {cmd}'
            )

    def _preload(self):
        """
        Load AF2 model weights in-process and build a StreamlinedColabFold
        instance for zero-overhead beam search scoring.

        Called automatically from __init__ when preload=True.
        """
        if self.device:
            device_id = int(self.device.split(':')[-1])
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

        self._configure_jax_runtime()
        data_dir = self._resolve_colabfold_data_dir()

        try:
            from colabfold.alphafold.models import load_models_and_params
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'In-process ColabFold requires the `colabfold` Python package to be '
                'installed in the active genie3 environment. '
                'Run scripts/setup/setup.sh to install it directly.'
            ) from exc

        use_templates = (self.mode == 'template')
        if self.mode == 'template':
            max_seq = 5
            max_extra_seq = 1
        elif self.mode == 'singleseq':
            max_seq = 1
            max_extra_seq = 1
        elif self.mode == 'msa':
            max_seq = 508
            max_extra_seq = 2048
        else:
            max_seq = 1
            max_extra_seq = 1

        logging.info(
            f'[ColabFoldHandler] Preloading AF2 weights '
            f'(mode={self.mode}, num_models={self.num_models}, '
            f'num_recycles={self.num_recycles})'
        )
        self._model_runner_and_params = load_models_and_params(
            num_models=self.num_models,
            use_templates=use_templates,
            num_recycles=self.num_recycles,
            num_ensemble=1,
            model_order=list(range(1, self.num_models + 1)),
            model_type='alphafold2_multimer_v3',
            data_dir=data_dir,
            stop_at_score=100,
            rank_by='multimer',
            use_dropout=False,
            max_seq=max_seq,
            max_extra_seq=max_extra_seq,
            use_cluster_profile=True,
            recycle_early_stop_tolerance=None,
            use_fuse=True,
            use_bfloat16=True,
            save_all=False,
            calc_extra_ptm=False,
        )

        from genie3.evaluation.model.fold.update_batch import StreamlinedColabFold
        self._streamlined = StreamlinedColabFold(
            model_runner_and_params=self._model_runner_and_params,
            model_type='alphafold2_multimer_v3',
            mode=self.mode,
            use_templates=use_templates,
            max_seq=max_seq,
            max_extra_seq=max_extra_seq,
        )
        logging.info('[ColabFoldHandler] AF2 weights loaded, StreamlinedColabFold ready')

    def _configure_jax_runtime(self):
        """
        Configure JAX GPU allocator behavior before importing ColabFold/JAX.

        Match colabfold.batch defaults as closely as possible unless the caller
        explicitly overrides them. This reduces the chance that the in-process
        path diverges from the stable subprocess/runtime behavior.
        """
        env_updates = {}
        if self.tf_force_unified_memory is not None:
            env_updates['TF_FORCE_UNIFIED_MEMORY'] = str(self.tf_force_unified_memory)
        if self.jax_mem_fraction is not None:
            env_updates['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.jax_mem_fraction)
        if self.jax_preallocate is not None:
            env_updates['XLA_PYTHON_CLIENT_PREALLOCATE'] = (
                'true' if self.jax_preallocate else 'false'
            )
        if self.jax_allocator:
            env_updates['XLA_PYTHON_CLIENT_ALLOCATOR'] = self.jax_allocator

        for key, value in env_updates.items():
            current = os.environ.get(key)
            if current != value:
                os.environ[key] = value

        if 'XDG_CACHE_HOME' not in os.environ:
            repo_root = Path(__file__).resolve().parents[5]
            fallback_cache = repo_root / 'packages' / '.cache' / 'xdg-cache'
            fallback_cache.mkdir(parents=True, exist_ok=True)
            os.environ['XDG_CACHE_HOME'] = str(fallback_cache)

        logging.info(
            '[ColabFoldHandler] JAX runtime configured '
            f'(tf_force_unified_memory={os.environ.get("TF_FORCE_UNIFIED_MEMORY", "unset")}, '
            f'preallocate={os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "unset")}, '
            f'allocator={os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "unset")}, '
            f'mem_fraction={os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "unset")})'
        )

    def _resolve_colabfold_data_dir(self):
        """
        Resolve the ColabFold parameter directory explicitly instead of relying
        on upstream cache discovery, which can drift to ~/.cache on clusters.
        """
        data_dir = os.environ.get('GENIE3_COLABFOLD_DATA_DIR')
        if data_dir:
            resolved = Path(data_dir)
        else:
            xdg_cache_home = os.environ.get('XDG_CACHE_HOME')
            if xdg_cache_home:
                resolved = Path(xdg_cache_home) / 'colabfold'
            else:
                repo_root = Path(__file__).resolve().parents[5]
                resolved = repo_root / 'packages' / '.cache' / 'xdg-cache' / 'colabfold'
                os.environ['XDG_CACHE_HOME'] = str(resolved.parent)

        resolved.mkdir(parents=True, exist_ok=True)
        os.environ['GENIE3_COLABFOLD_DATA_DIR'] = str(resolved)
        return resolved

    def predict_in_memory(
        self,
        binder_seqs,
        target_seq: str,
        template_path=None,
        target_msa_filepath=None,
        save_pdb_paths=None,
        max_binder_len=None,
    ):
        """
        Score K binder sequences in-process without any file I/O.

        Assumes target/template preparation has already happened via
        prepare_in_memory(). Prediction calls remain focused on scoring only.

        Args:
            binder_seqs: List[str] of K binder amino acid sequences.
            target_seq: Target amino acid sequence (shared across all beams).
            template_path: Pre-built pdb70 DB directory (or None).
            save_pdb_paths: Optional List[Optional[str]] of K PDB output paths.
            max_binder_len: Unused here. Kept for API compatibility.

        Returns:
            List[Dict] of K result dicts, each with keys: pae, plddt, ptm, iptm.
        """
        if self._streamlined is None:
            raise RuntimeError(
                '[ColabFoldHandler] predict_in_memory requires preload=True'
            )

        if (
            self._streamlined._cached_target_seq != target_seq
            or self._streamlined._cached_template_path != template_path
            or self._streamlined._cached_target_msa_filepath != target_msa_filepath
        ):
            raise RuntimeError(
                '[ColabFoldHandler] predict_in_memory requires prepare_in_memory() '
                'for the current target/template/MSA before scoring'
            )

        batch_start_time = time.perf_counter()
        results = []
        for k, binder_seq in enumerate(binder_seqs):
            save_path = (
                save_pdb_paths[k]
                if (save_pdb_paths and k < len(save_pdb_paths))
                else None
            )
            result = self._streamlined.predict(
                binder_seq, random_seed=k, save_pdb_path=save_path
            )
            results.append(result)
        batch_elapsed = time.perf_counter() - batch_start_time
        logging.info(
            '[ColabFoldHandler] predict_in_memory runtime '
            f'(n_seq={len(binder_seqs)}, target_len={len(target_seq)}) = '
            f'{batch_elapsed:.3f}s'
        )
        return results

    def prepare_in_memory(
        self,
        target_seq: str,
        template_path=None,
        target_msa_filepath=None,
        max_binder_len=None,
    ):
        """
        Prepare the in-process ColabFold runner for a specific target/template
        pair outside the per-sample scoring path.
        """
        if self._streamlined is None:
            raise RuntimeError(
                '[ColabFoldHandler] prepare_in_memory requires preload=True'
            )

        self._streamlined.prepare(
            target_seq,
            template_path=template_path,
            target_msa_filepath=target_msa_filepath,
        )

    def fold(self, structures_dir):
        """
        Run ColabFold to predict structures for all sequences in the input FASTA.

        Reads sequences from structures_dir/input.fasta, runs colabfold_batch,
        and organizes the output structure files into per-sample subdirectories.

        Args:
            structures_dir: Directory containing input.fasta and where output
                            structure files will be written
        """
        device_id = int(self.device.split(':')[-1])

        # Sanity check
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        if not os.path.exists(sequences_filepath):
            logging.error('Missing sequences filepath for template mode')
            exit(0)
        with open(sequences_filepath) as file:
            n_sequences = sum(1 for line in file if line.startswith('>'))
        logging.info(
            '[ColabFoldHandler] Folding started '
            f'(backend={self.backend}, mode={self.mode}, '
            f'sequences={n_sequences}, models={self.num_models}, '
            f'recycles={self.num_recycles})'
        )

        if self.backend == 'streamlined':
            if self._streamlined is None:
                logging.warning(
                    '[ColabFoldHandler] Streamlined backend requested without '
                    'preload=True; falling back to subprocess backend'
                )
            else:
                start_time = time.perf_counter()
                self._fold_streamlined(structures_dir)
                logging.info(
                    '[ColabFoldHandler] runtime profile '
                    f'backend=streamlined mode={self.mode} '
                    f'fold_seconds={time.perf_counter() - start_time:.3f}s'
                )
                return

        start_time = time.perf_counter()

        # Sanity check by mode
        if self.mode == 'singleseq':
            self._fold_single_seq(
                structures_dir=structures_dir,
                device_id=device_id
            )
        elif self.mode == 'msa':
            if self.version not in ['binder']:
                logging.error(f'Unsupported version to run with msa mode: {self.version}')
                exit(0)
            self._fold_msa(
                structures_dir=structures_dir,
                device_id=device_id
            )
        elif self.mode == 'template':
            if self.version not in ['binder']:
                logging.error(f'Unsupported version to run with msa mode: {self.version}')
                exit(0)
            self._fold_template(
                structures_dir=structures_dir,
                device_id=device_id
            )
        else:
            logging.error(f'Invalid folding mode: {self.mode}')
            exit(0)
        
        # Postprocess
        self._organize_structure_output(structures_dir)
        logging.info(
            '[ColabFoldHandler] runtime profile '
            f'backend=subprocess mode={self.mode} '
            f'fold_seconds={time.perf_counter() - start_time:.3f}s'
        )

    def _fold_streamlined(self, structures_dir):
        """
        Run in-process ColabFold and write reducer-compatible outputs.

        This keeps the existing reducer unchanged by writing ColabFold-shaped
        PDB/JSON artifacts into per-sample directories.
        """
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        sample_name_to_seq = parse_fasta(sequences_filepath)
        if len(sample_name_to_seq) == 0:
            logging.error('[ColabFoldHandler] No sequences found for streamlined fold')
            exit(0)

        target_seq = None
        target_msa_filepath = None
        template_path = None
        problem_info = None

        if self.version == 'binder':
            problem_name = parse_problem_name_from_fasta(sequences_filepath)
            with open(os.path.join(self.datadir, 'problems', f'{problem_name}.json')) as file:
                problem_info = json.load(file)

        # All designs in one input.fasta share the same target context.
        first_complex_seq = next(iter(sample_name_to_seq.values()))
        seq_parts = first_complex_seq.split(':')
        binder_seq = seq_parts[0]
        target_seq = ':'.join(seq_parts[1:]) if len(seq_parts) > 1 else ''

        if self.mode == 'msa':
            target_msa_filepath = problem_info['target_msa_filepath']
        elif self.mode == 'template':
            template_path = self._get_or_create_template_dir(
                structures_dir=structures_dir,
                target_filepath=problem_info['target_pdb_filepath'],
            )
        else:
            raise ValueError(
                f"[ColabFoldHandler] Unsupported mode '{self.mode}' for streamlined binder design. "
                "Use 'template' or 'msa'."
            )

        prepare_start_time = time.perf_counter()
        self.prepare_in_memory(
            target_seq=target_seq,
            template_path=template_path,
            target_msa_filepath=target_msa_filepath,
        )
        logging.info(
            '[ColabFoldHandler] runtime profile '
            f'backend=streamlined mode={self.mode} '
            f'prepare_in_memory_seconds={time.perf_counter() - prepare_start_time:.3f}s'
        )

        sample_records = []
        for original_idx, (sample_name, complex_seq) in enumerate(sample_name_to_seq.items()):
            complex_parts = complex_seq.split(':')
            sample_records.append({
                'original_idx': original_idx,
                'sample_name': sample_name,
                'complex_seq': complex_seq,
                'binder_seq': complex_parts[0],
                'total_length': sum(len(part) for part in complex_parts),
            })

        # Sort by total complex length to improve JAX/XLA trace reuse while
        # preserving deterministic seeds/output names via original_idx.
        sample_records.sort(key=lambda x: (x['total_length'], x['sample_name']))
        logging.info(
            '[ColabFoldHandler] streamlined scheduling by total length '
            + ' '.join([
                f'{record["sample_name"]}:{record["total_length"]}'
                for record in sample_records[:10]
            ])
            + (' ...' if len(sample_records) > 10 else '')
        )

        total_samples = len(sample_records)
        for sample_index, record in enumerate(sample_records, start=1):
            sample_idx = record['original_idx']
            sample_name = record['sample_name']
            complex_seq = record['complex_seq']
            parts = complex_seq.split(':')
            binder_seq = parts[0]
            sample_target_seq = ':'.join(parts[1:]) if len(parts) > 1 else ''
            if sample_target_seq != target_seq:
                raise ValueError(
                    '[ColabFoldHandler] Streamlined folding expects one shared target '
                    'sequence per batch'
                )

            sample_dir = os.path.join(structures_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            if total_samples <= 20 or sample_index == 1 or sample_index == total_samples or sample_index % 10 == 0:
                logging.info(
                    '[ColabFoldHandler] Folding sample '
                    f'{sample_index}/{total_samples} ({sample_name})'
                )
            sample_start_time = time.perf_counter()
            ranked_results = self._streamlined.predict_all(
                binder_seq=binder_seq,
                random_seed=sample_idx,
            )
            for result in ranked_results:
                model_name = self._normalize_model_name(
                    result.get('model_name', 'alphafold2_multimer_v3_model_1')
                )
                model_id = self._parse_model_id(model_name)
                basename = (
                    f'rank_{result["rank"]:03d}_{model_name}_seed_{sample_idx:03d}'
                )
                pdb_path = os.path.join(sample_dir, f'unrelaxed_{basename}.pdb')
                score_path = os.path.join(sample_dir, f'scores_{basename}.json')
                self._streamlined._write_prediction_pdb(
                    feature_dict=result['feature_dict'],
                    result=result['model_result'],
                    save_pdb_path=pdb_path,
                    binder_seq=binder_seq,
                    target_seq=target_seq,
                    remove_leading_feature_dimension=False,
                )
                self._write_streamlined_scores(
                    score_path=score_path,
                    result=result,
                    complex_seq=complex_seq,
                    model_id=model_id,
                )
            logging.info(
                '[ColabFoldHandler] runtime profile '
                f'backend=streamlined mode={self.mode} '
                f'sample={sample_name} n_ranked_models={len(ranked_results)} '
                f'sample_total_seconds={time.perf_counter() - sample_start_time:.3f}s'
            )

    def _get_or_create_template_dir(self, structures_dir, target_filepath):
        """
        Build the pdb70 template database once per evaluation batch.
        """
        from colabfold.batch import mk_hhsearch_db

        templates_dir = os.path.join(structures_dir, 'templates')
        marker = os.path.join(templates_dir, 'pdb70_a3m.ffdata')
        if os.path.exists(marker):
            return templates_dir

        os.makedirs(templates_dir, exist_ok=True)
        shutil.copyfile(target_filepath, os.path.join(templates_dir, 'temp.pdb'))
        mk_hhsearch_db(templates_dir)
        return templates_dir

    def _write_streamlined_scores(self, score_path, result, complex_seq, model_id):
        """
        Write a ColabFold-like scores JSON for reducer compatibility.
        """
        chain_ids = [chr(ord('A') + i) for i in range(len(complex_seq.split(':')))]
        per_chain_ptm = {
            chain_id: float(result.get('ptm', 0.0))
            for chain_id in chain_ids
        }
        payload = {
            'plddt': result['plddt'].tolist() if result.get('plddt') is not None else [],
            'pae': result['pae'].tolist() if result.get('pae') is not None else [],
            'ptm': float(result.get('ptm', 0.0)),
            'iptm': float(result.get('iptm', 0.0)),
            'actifptm': float(result.get('actifptm', result.get('iptm', 0.0))),
            'per_chain_ptm': result.get('per_chain_ptm', per_chain_ptm),
            'model_id': model_id,
            'ranking_score': float(result.get('ranking_score', result.get('iptm', 0.0))),
        }
        with open(score_path, 'w') as file:
            json.dump(payload, file)
        result.pop('feature_dict', None)
        result.pop('model_result', None)

    def _normalize_model_name(self, model_name):
        """
        Convert streamlined internal model names to ColabFold-style names.
        """
        if model_name.startswith('alphafold2_'):
            return model_name
        match = re.match(r'^model_(\d+)$', model_name)
        if match:
            return f'alphafold2_multimer_v3_model_{match.group(1)}'
        return model_name

    def _parse_model_id(self, model_name):
        normalized_model_name = self._normalize_model_name(model_name)
        match = re.search(r'_model_(\d+)$', normalized_model_name)
        if match:
            return int(match.group(1))
        return 1

    def _organize_structure_output(self, structures_dir):
        """
        Organize ColabFold output files into per-sample subdirectories.

        Reads sample names from input.fasta, creates a subdirectory for each,
        and moves all matching output files into the corresponding directory.

        Args:
            structures_dir: Directory containing ColabFold output files
        """

        # Load sample names
        sample_names = []
        with open(os.path.join(structures_dir, 'input.fasta')) as file:
            for line in file:
                if line.startswith('>'):
                    sample_names.append(line[1:].strip())

        # Organize output by samples
        for sample_name in sample_names:
            sample_structure_dir = os.path.join(structures_dir, sample_name)
            os.makedirs(sample_structure_dir)
            for filepath in glob.glob(os.path.join(structures_dir, f'{sample_name}_*.*')):
                elts = filepath.split(sample_name)
                filename = elts[1][1:] if elts[1].startswith('_') else sample_name + elts[1]
                output_filepath = os.path.join(sample_structure_dir, filename)
                shutil.move(filepath, output_filepath)

    def _fold_single_seq(self, structures_dir, device_id):
        """
        Run ColabFold in single-sequence mode (no MSA).

        Args:
            structures_dir: Directory containing input.fasta
            device_id: CUDA device index to use
        """

        # Define
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')

        # Predict
        cmd = ' '.join([
            f'CUDA_VISIBLE_DEVICES={device_id}',
            'colabfold_batch',
            sequences_filepath,
            structures_dir,
            '--msa-mode single_sequence',
            '--calc-extra-ptm',
            f'--num-models {self.num_models}',
            f'--num-recycle {self.num_recycles}',
        ])
        self._run_subprocess_command(cmd)

    def _fold_msa(self, structures_dir, device_id):
        """
        Run ColabFold with a complex MSA for binder design.

        Constructs a paired binder-target A3M MSA for each sequence in
        input.fasta and runs colabfold_batch on the resulting MSA files.

        Args:
            structures_dir: Directory containing input.fasta
            device_id: CUDA device index to use
        """

        ############################
        ###   MSA Construction   ###
        ############################

        # Load problem name
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        problem_name = parse_problem_name_from_fasta(sequences_filepath)
        with open(os.path.join(self.datadir, 'problems', f'{problem_name}.json')) as file:
            problem_info = json.load(file)
        
        # Create input directory
        input_dir = os.path.join(structures_dir, 'input')
        os.makedirs(input_dir)

        # Set up input directory
        sequences = parse_fasta(sequences_filepath)
        for name in sequences:
            make_complex_msa(
                target_msa_filepath=problem_info['target_msa_filepath'],
                binder_sequence=sequences[name].split(':')[0],
                output_filepath=os.path.join(input_dir, f'{name}.a3m')
            )
        
        ###################
        ###   Process   ###
        ###################

        cmd = ' '.join([
            f'CUDA_VISIBLE_DEVICES={device_id}',
            'colabfold_batch',
            input_dir,
            structures_dir,
            '--calc-extra-ptm',
            f'--num-models {self.num_models}',
            f'--num-recycle {self.num_recycles}',
        ])
        self._run_subprocess_command(cmd)

    def _fold_template(self, structures_dir, device_id):
        """
        Run ColabFold in single-sequence mode with a custom template.

        Copies the target PDB as a template and runs colabfold_batch with
        template-based folding for binder design.

        Args:
            structures_dir: Directory containing input.fasta
            device_id: CUDA device index to use
        """

        #################################
        ###   Template Construction   ###
        #################################

        # Load target pdb filepath
        sequences_filepath = os.path.join(structures_dir, 'input.fasta')
        problem_name = parse_problem_name_from_fasta(sequences_filepath)
        with open(os.path.join(self.datadir, 'problems', f'{problem_name}.json')) as file:
            problem_info = json.load(file)
        target_filepath = problem_info['target_pdb_filepath']

        # Create templates directory
        templates_dir = os.path.join(structures_dir, 'templates')
        os.makedirs(templates_dir)

        # Set up templates directory
        template_target_filepath = os.path.join(templates_dir, 'temp.pdb')
        shutil.copyfile(target_filepath, template_target_filepath)

        ###################
        ###   Process   ###
        ###################

        cmd = ' '.join([
            f'CUDA_VISIBLE_DEVICES={device_id}',
            'colabfold_batch',
            sequences_filepath,
            structures_dir,
            '--msa-mode single_sequence',
            '--templates',
            f'--custom-template-path {templates_dir}',
            '--calc-extra-ptm',
            f'--num-models {self.num_models}',
            f'--num-recycle {self.num_recycles}',
        ])
        self._run_subprocess_command(cmd)

    def _compile_filepath(self, design_filepath):
        """
        Parse name, domain, and file paths from a ColabFold design filepath.

        Args:
            design_filepath: Path to the predicted PDB or CIF file

        Returns:
            dict: 'name', 'domain', 'design_filepath', 'info_filepath'
        """
        sample_name = design_filepath.split('/')[-2]
        domain_name = sample_name.split('-')[0]
        resample_index = int(sample_name.split('-')[-1].split('_')[-1])
        info_filepath = design_filepath.replace('.pdb', '.json').replace('unrelaxed', 'scores')
        return {
            'name': sample_name,
            'domain': domain_name,
            'resample_id': resample_index,
            'design_filepath': design_filepath,
            'info_filepath': info_filepath
        }

    def _compile_rank(self, design_filepath):
        """
        Extract the rank of this prediction from the ColabFold filename.

        Args:
            design_filepath: Path to the predicted structure file

        Returns:
            dict: {'rank': int}
        """

        AF_FILENAME_RE = re.compile(
            r"""
            ^(?P<relax_state>unrelaxed|relaxed)             _rank_
            (?P<rank>\d{3})                                 _
            (?P<model_name>alphafold2_(?:ptm|multimer_v3))  _model_
            (?P<model_id>\d)                                _seed_
            (?P<seed>\d{3})$
            """,
            re.X,
        )

        name = Path(design_filepath.split('/')[-1]).stem        # drop .pdb or .pdb.gz extension
        m = AF_FILENAME_RE.match(name)
        if not m:
            raise ValueError(f'ColabFold filename {name} does not match expected pattern')
        d = m.groupdict()
        return {
            'model_name': d['model_name'],
            'rank': int(d['rank']),
            'model_id': int(d['model_id']),
            'seed': int(d['seed']),
            'relax_state': d['relax_state']
        }

    def _compile_confidence(self, info_filepath):
        """
        Extract pLDDT, pAE, pTM, ipTM, and per-chain pTM from a ColabFold JSON file.

        Args:
            info_filepath: Path to the JSON scores file

        Returns:
            dict: 'plddt', 'pae', 'ptm', 'iptm', 'per_chain_ptm'
        """
        with open(info_filepath) as file:
            info = json.load(file)
        return {
            'plddt': np.array(info['plddt']),
            'pae': np.array(info['pae']),
            'ptm': info['ptm'],
            'iptm': info['iptm'],
            'actifptm': info.get('actifptm', info.get('iptm', 0.0)),
            'per_chain_ptm': info.get('per_chain_ptm')
        }
    
    def _compile_ipsae(self, info_filepath, design_filepath, pae_cutoff=15, dist_cutoff=15):
        """
        Compute ipSAE interface quality metrics using the IPSAE tool.

        Args:
            info_filepath: Path to the AlphaFold JSON scores file
            design_filepath: Path to the predicted PDB structure file
            pae_cutoff: PAE cutoff for interface residue selection (default: 15)
            dist_cutoff: Distance cutoff for interface residue selection (default: 15)

        Returns:
            dict: 'ipsae', 'ipsae_d0dom', 'pdockq2' interface quality metrics
        """
        cmd = ' '.join([
            'python',
            'packages/IPSAE/ipsae.py',
            info_filepath,
            design_filepath,
            str(pae_cutoff),
            str(dist_cutoff)
        ])
        self._run_subprocess_command(cmd)
        output_filepath = design_filepath.replace('.pdb', f'_{pae_cutoff}_{dist_cutoff}.txt')
        df = pd.read_csv(
            output_filepath,
            sep=r"\s+",        # one or more spaces/tabs
            engine="python"    # needed for regex sep
        )
        df = df[(df['Chn1'] == 'A') & (df['Type'] == 'asym')]
        assert len(df) == 1
        return {
            'ipsae': df.iloc[0]['ipSAE'],
            'ipsae_d0dom': df.iloc[0]['ipSAE_d0dom'],
            'pdockq2': df.iloc[0]['pDockQ2'],
        }
