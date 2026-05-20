"""
Binder reducer for the Genie pipeline.

Compiles and filters results for binder protein design, computing
binding confidence, self-consistency RMSD, secondary structure content,
and target hotspot coverage metrics.
"""

import os
import glob
import copy
import json
import shutil
import logging
import time
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from collections import OrderedDict
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

from genie3.evaluation.model.fold.registry import get_fold_model
from genie3.evaluation.reducer.base import Reducer
from genie3.evaluation.utils.interface import check_binding_interface
from genie3.evaluation.utils.metric import compute_multimer_ca_rmsd, compute_ipsae
from genie3.evaluation.utils.parse import parse_fasta, parse_problem_name_from_fasta
from genie3.evaluation.utils.secondary import assign_secondary_structure


_BINDER_WORKER_CONTEXT = None


def _init_binder_worker(
    version,
    datadir,
    fold_model_name,
    fold_mode,
    sample_name_to_seq,
    problem_info,
    pdbs_dir,
):
    """
    Initialize one reducer worker process with shared read-only context.
    """
    global _BINDER_WORKER_CONTEXT
    _BINDER_WORKER_CONTEXT = {
        'fold_model': get_fold_model(
            name=fold_model_name,
            mode=fold_mode,
            version=version,
            datadir=datadir,
            preload=False,
        ),
        'sample_name_to_seq': sample_name_to_seq,
        'problem_info': problem_info,
        'pdbs_dir': pdbs_dir,
        'fold_model_name': fold_model_name,
    }


def _compile_binder_design(design_filepath):
    """
    Compile one folded binder design into one info row plus timing stats.
    """
    global _BINDER_WORKER_CONTEXT
    ctx = _BINDER_WORKER_CONTEXT

    timings = {
        'fold_compile_seconds': 0.0,
        'secondary_structure_seconds': 0.0,
        'consistency_seconds': 0.0,
        'interface_seconds': 0.0,
    }

    fold_compile_start_time = time.perf_counter()
    try:
        info = ctx['fold_model'].compile(design_filepath)
    except Exception as e:
        logging.warning(f'[Binder Reducer] Skipping {design_filepath} due to compilation error: {e}')
        return None, timings
    timings['fold_compile_seconds'] = time.perf_counter() - fold_compile_start_time

    info['model'] = ctx['fold_model_name']
    
    # Find the reference PDB by matching the base problem name.
    # The pdbs_dir should only contain the reference PDBs for this run.
    generation_filepath = None
    # We already parsed problem_info in _compile, but we need the base name here.
    # Instead of parsing it again, we can just look for ANY pdb in the dir since
    # there is usually only one, or we can use the domain as a prefix.
    base_prefix = info["domain"].split('-')[0] # e.g., csank1_175
    
    # Require an EXACT match to avoid 'csank4_1' matching 'csank4_100.pdb'
    expected_filename = f"{base_prefix}.pdb"
    if expected_filename in os.listdir(ctx['pdbs_dir']):
        generation_filepath = os.path.join(ctx['pdbs_dir'], expected_filename)
            
    if generation_filepath is None:
        # Fallback to the original logic if nothing was found
        generation_filepath = os.path.join(ctx['pdbs_dir'], f'{info["domain"]}.pdb')
        
    info['generation_filepath'] = generation_filepath

    seq = ctx['sample_name_to_seq'][info['name']]
    binder_seq = seq.strip().split(':')[0]
    target_seq = seq.strip().split(':')[1:]
    target_chain_ids = [
        chr(ord('A') + i + 1)
        for i in range(len(target_seq))
    ]
    binder_length = len(binder_seq)
    target_length = np.sum([
        len(_target_seq)
        for _target_seq in target_seq
    ])
    length = binder_length + target_length
    info.update({
        'len': length,
        'binder_len': binder_length,
        'binder_seq': binder_seq,
        'target_seq': target_seq,
    })

    assert len(info['plddt']) == length
    assert np.array(info['pae']).shape == (length, length)

    per_chain_ptm = info.get('per_chain_ptm')
    if per_chain_ptm is None:
        per_chain_ptm = {
            chain_id: info['ptm']
            for chain_id in ['A'] + target_chain_ids
        }
    info.update({
        'avg_plddt': np.mean(info['plddt']),
        'avg_binder_plddt': np.mean(info['plddt'][:binder_length]),
        'avg_target_plddt': np.mean(info['plddt'][-target_length:]),
        'avg_pae': np.mean(info['pae']),
        'avg_binder_pae': np.mean(info['pae'][:binder_length, :binder_length]),
        'avg_target_pae': np.mean(info['pae'][-target_length:, -target_length:]),
        'avg_interface_pae': 0.5 * (
            np.mean(info['pae'][:binder_length, -target_length:]) +
            np.mean(info['pae'][-target_length:, :binder_length])
        ),
        'min_interface_pae': min(
            np.min(info['pae'][:binder_length, -target_length:]),
            np.min(info['pae'][-target_length:, :binder_length])
        ),
        'binder_ptm': per_chain_ptm['A'],
        'target_ptm': np.mean([
            per_chain_ptm[target_chain_id]
            for target_chain_id in target_chain_ids
        ]),
        'per_chain_target_ptm': ':'.join([
            str(per_chain_ptm[target_chain_id])
            for target_chain_id in target_chain_ids
        ])
    })

    del info['plddt']
    del info['pae']
    info.pop('per_chain_ptm', None)

    secondary_structure_start_time = time.perf_counter()
    info.update(
        assign_secondary_structure(
            info['design_filepath'],
            chain='A',
            prefix='binder_'
        )
    )
    timings['secondary_structure_seconds'] = (
        time.perf_counter() - secondary_structure_start_time
    )

    use_chain_permutation = len(target_seq) > 1 and len(set(target_seq)) == 1

    consistency_start_time = time.perf_counter()
    binder_scrmsd, _, _ = compute_multimer_ca_rmsd(
        pdb1=info['generation_filepath'],
        pdb2=info['design_filepath'],
        chain_ids=['A'],
        anchor='A'
    )
    target_scrmsd, target_scrmsd_map, target_scrmsd_mode = compute_multimer_ca_rmsd(
        pdb1=info['generation_filepath'],
        pdb2=info['design_filepath'],
        chain_ids=target_chain_ids,
        permute=use_chain_permutation,
        anchor=None
    )
    complex_scrmsd, complex_scrmsd_map, complex_scrmsd_mode = compute_multimer_ca_rmsd(
        info['generation_filepath'],
        info['design_filepath'],
        chain_ids=['A'] + target_chain_ids,
        permute=use_chain_permutation,
        anchor='A'
    )
    timings['consistency_seconds'] = time.perf_counter() - consistency_start_time

    target_scrmsd_map_str = '|'.join([
        f'{key}->{target_scrmsd_map[key]}'
        for key in target_scrmsd_map
    ])
    complex_scrmsd_map_str = '|'.join([
        f'{key}->{complex_scrmsd_map[key]}'
        for key in complex_scrmsd_map
    ])
    info.update({
        'target_chain_permutation': use_chain_permutation,
        'binder_scrmsd': binder_scrmsd,
        'target_scrmsd': target_scrmsd,
        'target_scrmsd_map': target_scrmsd_map_str,
        'target_scrmsd_mode': target_scrmsd_mode,
        'complex_scrmsd': complex_scrmsd,
        'complex_scrmsd_map': complex_scrmsd_map_str,
        'complex_scrmsd_mode': complex_scrmsd_mode
    })

    target_hotspot_residue_indices = ctx['problem_info']['target_interface_residues']['hotspot']
    info.update({
        'n_target_hotspot': len(target_hotspot_residue_indices),
        'n_target_hotspot_hit': 0,
        'n_target_hotspot_near_hit': 0,
        'n_target_hotspot_miss': len(target_hotspot_residue_indices),
        'target_hotspot_coverage': 1.0 if len(target_hotspot_residue_indices) == 0 else 0.0,
    })
    if len(target_hotspot_residue_indices) > 0:
        interface_start_time = time.perf_counter()
        remapped_target_hotspot_residue_indices = []
        for target_hotspot_residue_index in target_hotspot_residue_indices:
            updated_chain_id = complex_scrmsd_map[target_hotspot_residue_index[0]]
            residue_index = int(target_hotspot_residue_index[1:])
            remapped_target_hotspot_residue_indices.append(f'{updated_chain_id}{residue_index}')
        interface_info = check_binding_interface(
            filepath=info['design_filepath'],
            hotspots=remapped_target_hotspot_residue_indices,
            binder_chain="A",
            cutoff=5.0,
            near_cutoff=6.5,
            heavy_only=True
        )
        timings['interface_seconds'] = time.perf_counter() - interface_start_time
        info.update({
            'n_target_hotspot': interface_info['overall']['n_hotspots_total'],
            'n_target_hotspot_hit': interface_info['overall']['n_hit_total'],
            'n_target_hotspot_near_hit': interface_info['overall']['n_near_total'],
            'n_target_hotspot_miss': (
                interface_info['overall']['n_hotspots_total'] -
                interface_info['overall']['n_hit_total'] -
                interface_info['overall']['n_near_total']
            ),
            'target_hotspot_coverage': interface_info['overall']['coverage']
        })

    return info, timings


class BinderReducer(Reducer):
    """Reducer for binder protein design evaluation."""

    def _validate(self):
        """
        Validate that the fold model and mode are supported for binder evaluation.
        """
        if not os.path.join(self.datadir):
            logging.error('[Binder Reducer] Data directory not provided')
            exit(0)
        if self.fold_model_name not in ['colabfold', 'boltz2', 'esmfold']:
            logging.error('[Binder Reducer] Available fold model: colabfold/boltz2/esmfold')
            exit(0)
        if self.fold_model_mode not in ['msa', 'template']:
            logging.error('[Binder Reducer] Available fold mode: msa/template')
            exit(0)

    def _compile(self, pdbs_dir, structures_dir, results_dir, verbose):
        """
        Compile per-design metrics for binder generation.

        For each folded complex structure, computes binding confidence scores
        (pLDDT, pAE, pTM), secondary structure content, self-consistency RMSD
        for binder, target, and complex chains, and target hotspot coverage.

        Args:
            pdbs_dir: Directory containing original generated PDB files
            structures_dir: Directory containing folded structure outputs
            results_dir: Directory to write compiled results
            verbose: Whether to display a progress bar
        """
        
        ##################
        ###   Set up   ###
        ##################

        # Load sequences
        sample_name_to_seq = parse_fasta(os.path.join(structures_dir, 'input.fasta'))

        # Load problem information
        problem_name = parse_problem_name_from_fasta(os.path.join(structures_dir, 'input.fasta'))
        with open(os.path.join(self.datadir, 'problems', f'{problem_name}.json')) as file:
            problem_info = json.load(file)

        ###################
        ###   Compile   ###
        ###################

        design_filepaths = []
        for dirpath in sorted(glob.glob(os.path.join(structures_dir, '*', ''))):
            dirname = os.path.basename(os.path.normpath(dirpath))
            if dirname == 'templates':
                continue
            design_filepaths.extend(sorted(
                glob.glob(os.path.join(dirpath, '*.pdb')) +
                glob.glob(os.path.join(dirpath, '*.cif'))
            ))

        if len(design_filepaths) == 0:
            logging.warning('[Binder Reducer] No folded structures found to compile')
            pd.DataFrame([]).to_csv(
                os.path.join(results_dir, 'info.csv'),
                index=False
            )
            return

        n_worker = min(
            len(design_filepaths),
            int(os.environ.get('GENIE3_BINDER_REDUCER_NPROC', max(1, os.cpu_count() or 1)))
        )
        logging.info(
            f'[Binder Reducer] compile scheduling n_design={len(design_filepaths)} '
            f'n_worker={n_worker}'
        )

        rows = []
        timing_sums = {
            'fold_compile_seconds': 0.0,
            'secondary_structure_seconds': 0.0,
            'consistency_seconds': 0.0,
            'interface_seconds': 0.0,
        }

        worker_items = []
        if n_worker <= 1:
            _init_binder_worker(
                version=self.version,
                datadir=self.datadir,
                fold_model_name=self.fold_model_name,
                fold_mode=self.fold_mode,
                sample_name_to_seq=sample_name_to_seq,
                problem_info=problem_info,
                pdbs_dir=pdbs_dir,
            )
            iterator = tqdm(
                design_filepaths,
                desc='Compiling', disable=not verbose
            )
            for design_filepath in iterator:
                worker_items.append(_compile_binder_design(design_filepath))
        else:
            with Pool(
                processes=n_worker,
                initializer=_init_binder_worker,
                initargs=(
                    self.version,
                    self.datadir,
                    self.fold_model_name,
                    self.fold_mode,
                    sample_name_to_seq,
                    problem_info,
                    pdbs_dir,
                ),
            ) as pool:
                iterator = pool.imap(_compile_binder_design, design_filepaths)
                worker_items = list(tqdm(
                    iterator,
                    total=len(design_filepaths),
                    desc='Compiling',
                    disable=not verbose,
                ))

        for info, timings in worker_items:
            if info is None:
                continue
            rows.append(info)
            for key in timing_sums:
                timing_sums[key] += timings.get(key, 0.0)

        rows.sort(key=lambda row: (
            row.get('name', ''),
            row.get('rank', 0),
            row.get('model_id', 0),
            row.get('design_filepath', ''),
        ))

        logging.info(
            '[Binder Reducer] compile timing totals '
            + ' '.join([
                f'{key}={value:.3f}s'
                for key, value in timing_sums.items()
            ])
        )
        if len(rows) > 0:
            logging.info(
                '[Binder Reducer] compile timing averages '
                + ' '.join([
                    f'{key}={value / len(rows):.3f}s'
                    for key, value in timing_sums.items()
                ])
            )

        # Save
        df = pd.DataFrame(rows)
        df.to_csv(
            os.path.join(results_dir, 'info.csv'),
            index=False
        )

    def _filter(self, results_dir):
        """
        Filter binder designs by model agreement and in-silico quality metrics.

        Applies version 0 filters:
        - Model agreement: complex RMSD < 2.5Å
        - In-silico binder quality: binder pTM > 0.8 and min interface pAE < 1.5Å

        Args:
            results_dir: Directory containing the compiled info.csv
        """
        
        ##################
        ###   Set up   ###
        ##################

        # Validate input files
        df_complex = pd.read_csv(os.path.join(results_dir, 'info.csv'))

        # Create output directory for successes based on version 0 filters
        v0_dir = os.path.join(results_dir, 'v0_success')
        if os.path.exists(v0_dir):
            logging.warning(f'Overwriting existing successes directory: {v0_dir}')
            shutil.rmtree(v0_dir)
        os.makedirs(os.path.join(v0_dir, 'successful_complexes'))
        os.makedirs(os.path.join(v0_dir, 'successful_incomplex_binders'))
        self._add_successes_dir(os.path.join(v0_dir, 'successful_incomplex_binders'))

        # # Create output directory for successes based on version 1 filters
        # v1_dir = os.path.join(results_dir, 'v1_success')
        # if os.path.exists(v1_dir):
        #     logging.error(f'Directory existed: {v1_dir}')
        #     exit(0)
        # os.makedirs(os.path.join(v1_dir, 'successful_complexes'))
        # os.makedirs(os.path.join(v1_dir, 'successful_incomplex_binders'))
        # self._add_successes_dir(os.path.join(v1_dir, 'successful_incomplex_binders'))

        ##################
        ###   Filter   ###
        ##################

        # Deduplicate
        df_complex = df_complex.sort_values(by='rank', ascending=True)
        df_complex = df_complex.drop_duplicates('name', keep='first')

        # Save general information
        with open(os.path.join(results_dir, 'log.txt'), 'w') as file:

            file.write('*** General ***\n')
            file.write('Number of generations: {}\n'.format(len(df_complex['domain'].unique())))
            file.write('\n\n')
            
            file.write('*** Version 0 Filters ***\n')
            file.write('==========\n')
            file.write('Model agreement: complex RMSD < 2.5Å\n')
            file.write('In-silico binder quality: binder pTM > 0.8 and min interface pAE < 1.5Å\n')
            file.write('Hotspot coverage: full coverage for <= 3 hotspots, >= 0.8 coverage otherwise\n')
            file.write('==========\n')
            file.write('Number of generations with high model agreement:          {}\n'.format(len(df_complex[df_complex['complex_scrmsd'] < 2.5]['domain'].unique())))
            file.write('Number of designs with high model agreement:              {}\n'.format(len(df_complex[df_complex['complex_scrmsd'] < 2.5]['name'].unique())))
            file.write('Number of generations with high in-silico binder quality: {}\n'.format(len(df_complex[(df_complex['binder_ptm'] > 0.8) & (df_complex['min_interface_pae'] < 1.5)]['domain'].unique())))
            file.write('Number of designs with high in-silico binder quality:     {}\n'.format(len(df_complex[(df_complex['binder_ptm'] > 0.8) & (df_complex['min_interface_pae'] < 1.5)]['name'].unique())))
            df_v0_success = df_complex[
                (df_complex['binder_ptm'] > 0.8) &
                (df_complex['min_interface_pae'] < 1.5) &
                (df_complex['complex_scrmsd'] < 2.5)
            ].copy()
            df_v0_success = df_v0_success[
                (
                    (df_v0_success['target_hotspot_coverage'] == 1)
                    & (df_v0_success['n_target_hotspot'] <= 3)
                ) | (
                    (df_v0_success['target_hotspot_coverage'] >= 0.8)
                    & (df_v0_success['n_target_hotspot'] > 3)
                )
            ].copy()
            file.write('Number of successful generations:                         {}\n'.format(len(df_v0_success['domain'].unique())))
            file.write('Number of successful designs:                             {}\n'.format(len(df_v0_success['name'].unique())))
            file.write('\n\n')

            # file.write('*** Version 1 Filters ***\n')
            # file.write('==========\n')
            # file.write('Model agreement: complex RMSD < 2.5Å\n')
            # file.write('In-silico binder quality: pQockQ2 > 0.80, ipTM_af > 0.85 and ipSAE_d0dom > 0.65\n')
            # file.write('==========\n')
            # file.write('Number of generations with high model agreement:          {}\n'.format(len(df_complex[df_complex['complex_scrmsd'] < 2.5]['domain'].unique())))
            # file.write('Number of designs with high model agreement:              {}\n'.format(len(df_complex[df_complex['complex_scrmsd'] < 2.5]['name'].unique())))
            # file.write('Number of generations with high in-silico binder quality: {}\n'.format(len(df_complex[(df_complex['pdockq2'] > 0.8) & (df_complex['iptm'] > 0.85) & (df_complex['ipsae_d0dom'] > 0.65)]['domain'].unique())))
            # file.write('Number of designs with high in-silico binder quality:     {}\n'.format(len(df_complex[(df_complex['pdockq2'] > 0.8) & (df_complex['iptm'] > 0.85) & (df_complex['ipsae_d0dom'] > 0.65)]['name'].unique())))
            # df_v1_success = df_complex[
            #     (df_complex['pdockq2'] > 0.8) & 
            #     (df_complex['iptm'] > 0.85) & 
            #     (df_complex['ipsae_d0dom'] > 0.65) &
            #     (df_complex['complex_scrmsd'] < 2.5)
            # ].copy()
            # file.write('Number of successful generations:                         {}\n'.format(len(df_v1_success['domain'].unique())))
            # file.write('Number of successful designs:                             {}\n'.format(len(df_v1_success['name'].unique())))
            # file.write('\n\n')

        # Save
        for df_success, outdir in [
            (df_v0_success, v0_dir),
            # (df_v1_success, v1_dir)
        ]:

            # Save
            df_success.to_csv(os.path.join(outdir, 'success_info.csv'), index=False)

            # Extract PDB files
            for _, row in df_success.iterrows():

                # Copy complex pdb file
                file_ext = row['design_filepath'].split('.')[-1]
                assert file_ext in ['pdb', 'cif'], 'Invalid design filepath extension'
                shutil.copyfile(
                    row['design_filepath'],
                    os.path.join(outdir, 'successful_complexes', f'{row["name"]}.{file_ext}')
                )

                # Save incomplex binder pdb file
                extract_chain(
                    in_path=row['design_filepath'],
                    out_path=os.path.join(outdir, 'successful_incomplex_binders', f'{row["name"]}.pdb')
                )


class _ChainSelect(Select):
    """
    Biopython PDBIO Select subclass that filters to a single chain.

    Used internally to extract a specific chain from a multi-chain PDB file.
    """

    def __init__(self, wanted_chain_id):
        """
        Args:
            wanted_chain_id: Chain ID to keep (all other chains are excluded)
        """
        super().__init__()
        self.wanted_chain_id = wanted_chain_id

    def accept_chain(self, chain):
        """
        Return True only for the chain matching wanted_chain_id.

        Args:
            chain: Biopython Chain object

        Returns:
            bool: True if chain.id == wanted_chain_id
        """
        return chain.id == self.wanted_chain_id


def extract_chain(in_path, out_path, chain_id="A"):
    """
    Extract a single chain from a PDB or mmCIF file and save it as a PDB.

    Parameters
    ----------
    in_path : str
        Path to input .pdb or .cif/.mmcif file.
    chain_id : str, optional
        Chain identifier to extract (default: "A").
    out_path : str or None, optional
        Output PDB path. If None, uses '<input_stem>_chain<id>.pdb'.

    Returns
    -------
    str
        Path to the written PDB file containing only the requested chain.
    """
    in_path = os.path.abspath(in_path)
    ext = os.path.splitext(in_path)[1].lower()

    # Choose parser based on extension
    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
    elif ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' (expected .pdb or .cif/.mmcif)")

    structure = parser.get_structure("struct", in_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, select=_ChainSelect(chain_id))

    return out_path
