"""
Pipeline runner orchestrating the full evaluation workflow.

Coordinates sanitization, splitting, mapping, aggregation, and reduction
stages across one or more GPU devices.
"""

import os
import glob
import json
import shutil
import logging
import sys
import time
from tqdm import tqdm
from multiprocessing import Process

from genie3.evaluation.mapper import Mapper
from genie3.evaluation.reducer.registry import get_reducer
from genie3.runtime.progress import get_active_reporter


class Runner:
    """Orchestrates the full evaluation pipeline across multiple stages."""

    def _estimate_pdb_weight(self, pdb_filepath):
        """
        Estimate job cost from the number of unique residues in a PDB.

        This is used only for device assignment, so it should be cheap,
        deterministic, and correlated with downstream runtime.
        """
        residues = set()
        with open(pdb_filepath) as file:
            for line in file:
                if not line.startswith('ATOM'):
                    continue
                residues.add((line[21], line[22:26], line[26]))
        return len(residues)

    def evaluate(self, version, rootdir, n_device, verbose, skip_reduce=False, **kwargs):
        """
        Run the full evaluation pipeline end-to-end.

        Executes sanitize → split → map → aggregate → reduce in sequence.

        Args:
            version: Pipeline version string (e.g. 'binder', 'scaffold')
            rootdir: Root directory containing the 'pdbs' subdirectory
            n_device: Number of GPU devices to use in parallel
            verbose: Whether to log progress
            skip_reduce: Whether to skip the final reduce stage
            **kwargs: Additional parameters forwarded to map and reduce stages
        """
        timings = {}

        get_active_reporter().set_status("sanitize")
        start_time = time.perf_counter()
        self.sanitize(rootdir, verbose)
        timings['sanitize_seconds'] = time.perf_counter() - start_time

        get_active_reporter().set_status("split")
        start_time = time.perf_counter()
        self.split(rootdir, n_device, verbose)
        timings['split_seconds'] = time.perf_counter() - start_time

        get_active_reporter().set_status("map")
        start_time = time.perf_counter()
        self.map(version, rootdir, n_device, verbose, kwargs)
        timings['map_seconds'] = time.perf_counter() - start_time

        get_active_reporter().set_status("aggregate")
        start_time = time.perf_counter()
        map_profile = self.aggregate(rootdir, verbose)
        timings['aggregate_seconds'] = time.perf_counter() - start_time

        if skip_reduce:
            logging.info('[Runner] Skipping reduce stage')
        else:
            get_active_reporter().set_status("reduce")
            start_time = time.perf_counter()
            self.reduce(version, rootdir, verbose, kwargs)
            timings['reduce_seconds'] = time.perf_counter() - start_time

        get_active_reporter().set_status("done")

        summary = dict(timings)
        summary.update(map_profile)
        with open(os.path.join(rootdir, 'evaluation_map_stats.json'), 'w') as file:
            json.dump(summary, file, indent=2, sort_keys=True)
            file.write('\n')

        logging.info(
            '[Runner] runtime profile '
            + ' '.join([
                f'{key}={value:.3f}s'
                for key, value in timings.items()
            ])
        )

    def sanitize(self, rootdir, verbose):
        """
        Sanitize PDB files in-place by normalizing chain IDs, residue indices,
        occupancy, and B-factors.

        Args:
            rootdir: Root directory containing the 'pdbs' subdirectory
            verbose: Whether to display a progress bar
        """

        ##################
        ###   Set up   ###
        ##################

        # Validate input directory
        pdbs_dir = os.path.join(rootdir, 'pdbs')
        if not os.path.exists(pdbs_dir):
            logging.error(f'Missing directory for {pdbs_dir}')
            exit(0)

        ###################
        ###   Process   ###
        ###################

        for pdb_filepath in tqdm(
            glob.glob(os.path.join(pdbs_dir, '*.pdb')),
            desc='Sanitizing', disable=not verbose
        ):
            lines = []
            occ_default = 1.00
            bfac_default = 20.00
            chain_id_map = {}
            offset_by_chain = {}

            with open(pdb_filepath) as file:
                for line in file:
                    if not line.startswith('ATOM'):
                        continue
                    line = line.ljust(80)

                    # Parse chain and residue information
                    orig_chain = line[21]
                    orig_resi = int(line[22:26])
                    icode = line[26]

                    # Remap chain id and store residue index offset
                    if orig_chain not in chain_id_map:
                        chain_id_map[orig_chain] = chr(ord('A') + len(chain_id_map))
                    ch = chain_id_map[orig_chain]
                    if ch not in offset_by_chain:
                        offset_by_chain[ch] = 1 - orig_resi

                    # Recompute residue index
                    resi = orig_resi + offset_by_chain[ch]

                    # Create new line
                    new_resseq_str = f"{resi:>4}"
                    line = line[:21] + ch + new_resseq_str + line[26:]

                    # Overwrite occupancy and b-factor
                    occ = line[54:60].strip()
                    bfac = line[60:66].strip()
                    occ  = f"{float(occ) if occ else occ_default:6.2f}"
                    bfac = f"{float(bfac) if bfac else bfac_default:6.2f}"
                    line = line[:54] + occ + bfac + line[66:]

                    # Store
                    lines.append(line.rstrip())

            # Save sanitized PDB
            with open(pdb_filepath, 'w') as file:
                file.write('\n'.join(lines) + '\n')

    def split(self, rootdir, n_device, verbose):
        """
        Distribute PDB files across per-device subdirectories.

        Args:
            rootdir: Root directory containing the 'pdbs' subdirectory
            n_device: Number of GPU devices to split across
            verbose: Whether to display a progress bar
        """

        ##################
        ###   Set up   ###
        ##################

        # Validate input directory
        pdbs_dir = os.path.join(rootdir, 'pdbs')
        if not os.path.exists(pdbs_dir):
            logging.error(f'Missing directory for {pdbs_dir}')
            exit(0)

        # Create output directory
        devices_dir = os.path.join(rootdir, 'devices')
        if os.path.exists(devices_dir):
            logging.warning(f'Removing stale devices directory: {devices_dir}')
            shutil.rmtree(devices_dir)
        for device_id in range(n_device):
            os.makedirs(os.path.join(devices_dir, f'device_{device_id}', 'pdbs'))

        ###################
        ###   Process   ###
        ###################

        jobs = []
        for pdb_filepath in glob.glob(os.path.join(pdbs_dir, '*.pdb')):
            jobs.append((pdb_filepath, self._estimate_pdb_weight(pdb_filepath)))

        jobs.sort(key=lambda x: (-x[1], x[0]))

        device_loads = [0] * n_device
        device_jobs = [[] for _ in range(n_device)]
        for pdb_filepath, weight in jobs:
            device_id = min(range(n_device), key=lambda d: (device_loads[d], d))
            device_jobs[device_id].append((pdb_filepath, weight))
            device_loads[device_id] += weight

        for device_id, jobs_for_device in enumerate(device_jobs):
            for pdb_filepath, _weight in tqdm(
                jobs_for_device,
                desc=f'Splitting device_{device_id}',
                disable=not verbose
            ):
                filename = pdb_filepath.split('/')[-1]
                output_filepath = os.path.join(
                    devices_dir, f'device_{device_id}', 'pdbs', filename
                )
                shutil.copyfile(pdb_filepath, output_filepath)

        logging.info(
            '[Runner.split] device load estimate '
            + ' '.join([
                f'device_{device_id}={load}'
                for device_id, load in enumerate(device_loads)
            ])
        )

    def map(self, version, rootdir, n_device, verbose, params):
        """
        Run inverse folding and structure prediction in parallel across devices.

        Spawns one process per device, each running a Mapper on its assigned
        PDB subset.

        Args:
            version: Pipeline version string
            rootdir: Root directory containing the 'devices' subdirectory
            n_device: Number of GPU devices
            verbose: Whether to log progress
            params: Dictionary of model configuration parameters
        """

        ##################
        ###   Set up   ###
        ##################

        def map_by_device(version, rootdir, device, verbose, params):
            """
            Run the Mapper for a single device's PDB subset.

            Args:
                version: Pipeline version string
                rootdir: Per-device root directory (contains 'pdbs' subdirectory)
                device: CUDA device string (e.g. 'cuda:0')
                verbose: Whether to display progress bars
                params: Additional mapper parameters
            """
            workers_dir = os.environ.get("GENIE3_WORKERS_DIR")
            worker_name = device.replace(":", "_")
            primary_worker = device.endswith(":0")
            worker_log_path = None
            stream = None
            original_stdout = None
            original_stderr = None
            root_logger = logging.getLogger()
            previous_handlers = list(root_logger.handlers)
            previous_level = root_logger.level
            if workers_dir:
                os.makedirs(workers_dir, exist_ok=True)
                worker_log_path = os.path.join(
                    workers_dir,
                    f"evaluate_{worker_name}.log",
                )
                handlers = [logging.FileHandler(worker_log_path)]
                if verbose and primary_worker:
                    handlers.append(logging.StreamHandler(sys.stdout))
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                for handler in handlers:
                    handler.setFormatter(formatter)
                root_logger.handlers = handlers
                root_logger.setLevel(logging.DEBUG)
            elif verbose and primary_worker:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)],
                    force=True,
                )

            original_fd1 = None
            original_fd2 = None
            if worker_log_path is not None and not verbose:
                stream = open(worker_log_path, "a")
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = stream
                sys.stderr = stream
                # Also redirect OS-level fds so output from C extensions
                # (e.g. DeepSpeed linker errors) goes to the log file.
                original_fd1 = os.dup(1)
                original_fd2 = os.dup(2)
                os.dup2(stream.fileno(), 1)
                os.dup2(stream.fileno(), 2)

            try:
                import torch
                if torch.cuda.is_available():
                    torch.set_float32_matmul_precision("high")
            except ImportError:
                pass

            try:
                logging.info(
                    '[%s] Worker started (rootdir=%s, inverse_fold_model=%s, '
                    'fold_model=%s, fold_mode=%s)',
                    device,
                    rootdir,
                    params.get('inverse_fold_model_name'),
                    params.get('fold_model_name'),
                    params.get('fold_mode'),
                )
                mapper = Mapper(
                    version=version,
                    datadir=params['datadir'],
                    device=device,
                    inverse_fold_mode=params['inverse_fold_mode'],
                    inverse_fold_model_name=params['inverse_fold_model_name'],
                    inverse_fold_num_seq=params['inverse_fold_num_seq'],
                    fold_mode=params['fold_mode'],
                    fold_model_name=params['fold_model_name'],
                    fold_backend=params.get('fold_backend', 'streamlined'),
                    fold_num_models=params.get('fold_num_models', 5),
                    fold_num_recycles=params.get('fold_num_recycles', 20),
                    verbose=verbose,
                )
                mapper.map(rootdir, verbose)
                logging.info('[%s] Worker completed', device)
            finally:
                if stream is not None:
                    stream.flush()
                    if original_fd1 is not None:
                        os.dup2(original_fd1, 1)
                        os.close(original_fd1)
                    if original_fd2 is not None:
                        os.dup2(original_fd2, 2)
                        os.close(original_fd2)
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    stream.close()
                if worker_log_path is not None:
                    for handler in root_logger.handlers:
                        handler.close()
                    root_logger.handlers = previous_handlers
                    root_logger.setLevel(previous_level)

        # Validate input directory
        devices_dir = os.path.join(rootdir, 'devices')
        if not os.path.exists(devices_dir):
            logging.error(f'Missing directory for {devices_dir}')
            exit(0)

        ###################
        ###   Process   ###
        ###################

        processes = []
        try:
            for device_id in range(n_device):
                device_rootdir = os.path.join(devices_dir, f'device_{device_id}')
                if not os.path.exists(os.path.join(device_rootdir, 'pdbs')):
                    continue
                process = Process(
                    target=map_by_device,
                    args=(
                        version,
                        device_rootdir,
                        f'cuda:{device_id}',
                        verbose,
                        params,
                    ),
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
        except KeyboardInterrupt:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            raise

        for process in processes:
            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f'Evaluation worker exited with code {process.exitcode}'
                )

    def aggregate(self, rootdir, verbose):
        """
        Merge per-device sequence and structure outputs into root-level directories.

        Args:
            rootdir: Root directory containing the 'devices' subdirectory
            verbose: Whether to display progress bars
        """

        ##################
        ###   Set up   ###
        ##################

        # Validate input directory
        devices_dir = os.path.join(rootdir, 'devices')
        if not os.path.exists(devices_dir):
            logging.error(f'Missing directory for {devices_dir}')
            exit(0)

        # Create output directories
        sequences_dir = os.path.join(rootdir, 'sequences')
        structures_dir = os.path.join(rootdir, 'structures')
        if os.path.exists(sequences_dir):
            logging.error(f'Directory existed: {sequences_dir}')
            exit(0)
        if os.path.exists(structures_dir):
            logging.error(f'Directory existed: {structures_dir}')
            exit(0)
        os.makedirs(sequences_dir)
        os.makedirs(structures_dir)

        ###################
        ###   Process   ###
        ###################

        profile = {
            'worker_count': 0,
            'generated_sequences': 0,
            'predicted_structures': 0,
            'representative_device': None,
            'representative_generated_sequences': 0,
            'representative_predicted_structures': 0,
            'inverse_fold_seconds': 0.0,
            'fold_seconds': 0.0,
            'inverse_fold_model_init_seconds': 0.0,
            'fold_model_init_seconds': 0.0,
            'inverse_fold_wall_seconds': 0.0,
            'fold_wall_seconds': 0.0,
            'inverse_fold_model_init_wall_seconds': 0.0,
            'fold_model_init_wall_seconds': 0.0,
            'fold_model_name': None,
            'fold_mode': None,
            'fold_num_models': None,
            'fold_num_recycles': None,
        }

        for dirpath in tqdm(
            glob.glob(os.path.join(devices_dir, 'device_*', '')),
            desc='Aggregating', disable=not verbose
        ):
            profile_path = os.path.join(dirpath, 'evaluation_map_stats.json')
            if not os.path.exists(profile_path):
                raise FileNotFoundError(
                    f'Missing evaluation runtime profile in {dirpath}; '
                    'expected evaluation_map_stats.json'
                )
            with open(profile_path) as file:
                device_profile = json.load(file)
            profile['worker_count'] += 1
            device_name = os.path.basename(os.path.normpath(dirpath)).replace('device_', 'cuda:')
            if (
                profile['representative_device'] is None or
                device_name == 'cuda:0'
            ):
                profile['representative_device'] = device_name
                profile['representative_generated_sequences'] = int(
                    device_profile.get('generated_sequences', 0)
                )
                profile['representative_predicted_structures'] = int(
                    device_profile.get('predicted_structures', 0)
                )
            for key in profile:
                if key in {
                    'representative_device',
                    'fold_model_name',
                    'fold_mode',
                    'fold_num_models',
                    'fold_num_recycles',
                }:
                    if profile[key] is None:
                        profile[key] = device_profile.get(key)
                elif key in {
                    'representative_generated_sequences',
                    'representative_predicted_structures',
                }:
                    representative_device = profile.get('representative_device')
                    if representative_device == device_name:
                        source_key = key.removeprefix('representative_')
                        profile[key] = int(device_profile.get(source_key, 0))
                elif key in {
                    'inverse_fold_wall_seconds',
                    'fold_wall_seconds',
                    'inverse_fold_model_init_wall_seconds',
                    'fold_model_init_wall_seconds',
                }:
                    representative_device = profile.get('representative_device')
                    if representative_device == device_name:
                        source_key = key.removesuffix('_wall_seconds') + '_seconds'
                        profile[key] = float(
                            device_profile.get(key, device_profile.get(source_key, 0.0))
                        )
                else:
                    profile[key] += device_profile.get(key, 0.0)

            for filepath in glob.glob(os.path.join(dirpath, 'sequences', '*.fasta')):
                filename = filepath.split('/')[-1]
                shutil.copyfile(filepath, os.path.join(sequences_dir, filename))

            sample_names: set[str] = set()
            input_fasta_path = os.path.join(dirpath, 'structures', 'input.fasta')
            if os.path.exists(input_fasta_path):
                lines = []
                with open(input_fasta_path) as file:
                    for line in file:
                        lines.append(line.strip())
                        if line.startswith('>'):
                            sample_names.add(line[1:].strip())
                output_fasta_path = os.path.join(structures_dir, 'input.fasta')
                if not os.path.exists(output_fasta_path):
                    with open(output_fasta_path, 'w') as file:
                        file.write('\n'.join(lines) + '\n')
                else:
                    with open(output_fasta_path, 'a') as file:
                        file.write('\n'.join(lines) + '\n')

            for filepath in glob.glob(os.path.join(dirpath, 'structures', '*')):
                filename = filepath.split('/')[-1]
                if filename == 'input.fasta':
                    continue
                src = filepath
                dst = os.path.join(structures_dir, filename)
                if os.path.isdir(src):
                    if filename not in sample_names:
                        continue
                    shutil.copytree(src, dst)
                else:
                    shutil.copyfile(src, dst)

        shutil.rmtree(devices_dir)
        return profile

    def reduce(self, version, rootdir, verbose, params):
        """
        Compile aggregated outputs into metrics/results and optionally novelty search.

        Args:
            version: Pipeline version string
            rootdir: Root directory containing 'pdbs', 'structures', and 'sequences'
            verbose: Whether to display progress bars
            params: Reducer parameters (datadir, calc_novelty, fold model config, etc.)
        """

        reducer = get_reducer(
            version=version,
            datadir=params['datadir'],
            fold_model_name=params['fold_model_name'],
            fold_mode=params['fold_mode'],
            calc_novelty=params['calc_novelty'],
        )
        reducer.reduce(rootdir=rootdir, verbose=verbose)
