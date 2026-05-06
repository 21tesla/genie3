"""
ColabFold-based reward for beam search.

Scores beam candidates by running ColabFold structure prediction on the
model's current sequence prediction (from ai logits) and returning the
ipTM score as the reward signal.
"""

import os
import json
import glob
import shutil
import time
import torch
import tempfile
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from ml_collections import ConfigDict

from genie3.generation.diffusion.reward.base import Reward
from genie3.generation.np.protein_constants import PROTEIN_RESTYPES, RESTYPE_1_TO_3
from genie3.generation.utils.feat_utils import debatchify_tensor_features, save_np_features_to_pdb
from genie3.evaluation.utils.parse import parse_fasta


class ColabFoldReward(Reward):
    """
    Reward function that uses ColabFold ipTM as the beam quality signal.

    At each scoring checkpoint, this reward:
    1. Decodes the model's current sequence prediction from `out['ai']` logits
       (argmax of softmax).
    2. Constructs a per-beam FASTA file (binder:target format for multimer).
    3. Runs ColabFold in single-sequence mode (fast; no MSA) on all beams
       simultaneously.
    4. Parses `iptm` from each beam's scores JSON and returns as reward tensor.

    Requires `predict_sequence=True` in the sampler config so that `out['ai']`
    is populated by the denoiser during intermediate steps.
    """

    # Map from residue index to single-letter code
    _IDX_TO_AA = {i: aa for i, aa in enumerate(PROTEIN_RESTYPES)}

    def __init__(self, config: ConfigDict):
        """
        Initialize ColabFoldReward and load the ColabFoldHandler.

        Args:
            config: ConfigDict with fields:
                version  (str): Pipeline version, e.g. 'binder'
                mode     (str): ColabFold mode — use 'singleseq' for fast scoring
                device   (str): CUDA device string, e.g. 'cuda:0'
                datadir  (str): Path to the problem data directory
        """
        super().__init__(config)

        from genie3.evaluation.model.fold.colabfold import ColabFoldHandler
        self.handler = ColabFoldHandler(
            version=config.version,
            mode=config.mode,
            device=config.device,
            datadir=config.datadir,
            backend=config.get("backend", "streamlined"),
            num_models=config.num_models,
            num_recycles=config.num_recycles,
            tf_force_unified_memory=config.get("tf_force_unified_memory", "1"),
            jax_preallocate=config.get("jax_preallocate", None),
            jax_mem_fraction=config.get("jax_mem_fraction", "4.0"),
            jax_allocator=config.get("jax_allocator", None),
            preload=(config.get("backend", "streamlined") == "streamlined"),
            verbose=bool(config.get("verbose", False)),
        )

        self.mpnn = None
        if config.get("use_proteinmpnn", False):
            from genie3.evaluation.model.inverse_fold.proteinmpnn import ProteinMPNNHanlder
            self.mpnn = ProteinMPNNHanlder(
                version=config.version,
                mode=config.mode,
                device=config.device,
                num_samples=config.proteinmpnn_num_samples,
                sampling_temperature=config.proteinmpnn_sampling_temperature,
            )

    def _get_or_create_template_dir(
        self, problem_name: str, problem_info: Optional[dict]
    ) -> Optional[str]:
        """
        Return a persistent pdb70 DB directory for the target PDB.

        Creates the directory (and runs mk_hhsearch_db) at most once per
        problem — the marker file pdb70_a3m.ffdata is used to detect an
        already-built DB.  The directory lives alongside the problem JSON so
        it survives across scoring checkpoints.

        Returns None if mode is not 'template' or target PDB is unavailable.
        """
        if self.config.mode != 'template':
            return None
        if not problem_info:
            return None

        from colabfold.batch import mk_hhsearch_db

        target_pdb = problem_info.get('target_pdb_filepath') or problem_info.get('target_filepath')
        if not target_pdb or not os.path.exists(target_pdb):
            logging.warning(
                f'[ColabFoldReward] target PDB not found for {problem_name}, '
                'falling back to mock templates'
            )
            return None

        problems_dir = os.path.dirname(os.path.abspath(target_pdb))
        template_dir = os.path.join(problems_dir, f'.cf_templates_{problem_name}')
        marker = os.path.join(template_dir, 'pdb70_a3m.ffdata')

        if not os.path.exists(marker):
            os.makedirs(template_dir, exist_ok=True)
            import shutil as _shutil
            _shutil.copyfile(target_pdb, os.path.join(template_dir, 'temp.pdb'))
            mk_hhsearch_db(template_dir)
            logging.info(f'[ColabFoldReward] Built pdb70 DB at {template_dir}')

        return template_dir

    def _make_pdb_paths(
        self, job_dirs: Optional[List[str]], K: int
    ) -> Optional[List[Optional[str]]]:
        """Build per-beam PDB output paths from job_dirs, or return None."""
        if not job_dirs:
            return None
        return [
            os.path.join(job_dirs[k], 'predicted_structure.pdb')
            if (k < len(job_dirs) and job_dirs[k])
            else None
            for k in range(K)
        ]

    def _load_problem_context(
        self, problem_name: Optional[str]
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Load target sequence and raw problem JSON for a given problem name.
        """
        if not problem_name:
            return None, None

        json_path = os.path.join(self.config.datadir, 'problems', f'{problem_name}.json')
        try:
            with open(json_path) as f:
                problem_info = json.load(f)
            fasta_path = problem_info.get('target_fasta_filepath')
            if fasta_path:
                sequences = parse_fasta(fasta_path)
                target_seq = next(iter(sequences.values())) if sequences else None
            else:
                target_seq = None
            return target_seq, problem_info
        except Exception as e:
            logging.warning(f'[ColabFoldReward] Could not read target sequence: {e}')
            return None, None

    def prime_problem(self, problem_name: Optional[str]):
        """
        Prepare the in-process ColabFold runner for one problem.
        Safe to call repeatedly; the handler caches target/template state.
        """
        if self.handler._streamlined is None:
            return
        if self.config.mode not in ('template', 'singleseq', 'msa'):
            return

        target_seq, problem_info = self._load_problem_context(problem_name)
        if target_seq is None:
            return

        prime_start_time = time.perf_counter()
        template_path = self._get_or_create_template_dir(problem_name, problem_info)
        target_msa_filepath = (
            problem_info.get('target_msa_filepath') if problem_info is not None else None
        )
        if self.config.mode != 'msa' or target_msa_filepath is not None:
            self.handler.prepare_in_memory(
                target_seq=target_seq,
                template_path=template_path,
                target_msa_filepath=target_msa_filepath,
            )
        prime_elapsed = time.perf_counter() - prime_start_time
        logging.info(
            f'[ColabFoldReward] prime_problem runtime ({problem_name}) = {prime_elapsed:.3f}s'
        )

    def compute(
        self,
        batch: Dict[str, torch.Tensor],
        xl: torch.Tensor,
        out: Dict[str, torch.Tensor],
        job_dirs: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Score K beam candidates using ColabFold ipAE.

        Args:
            batch: Tiled conditioning batch. Shape of each tensor: [K, ...]
                   Must include 'gt_atom_mask' [K, N] and optionally
                   'target_sequence' (str) from problem config.
            xl: Denoised coordinates at current step. Shape: [K, N, 3]
            out: Model output dict; must contain 'ai' [K, N, n_aa] logits.
            job_dirs: Optional list of K directory paths. When provided for
                beam k, ColabFold PDBs and colabfold_scores.json are copied
                into job_dirs[k].

        Returns:
            Tuple of:
                scores   — reward per beam (-i_pae), shape [K]. Higher = better.
                metrics  — list of K dicts with keys: i_pae, iptm, ptm, plddt.
        """
        K = xl.shape[0]

        if 'ai' not in out and self.mpnn is None:
            logging.warning(
                '[ColabFoldReward] out["ai"] not found — returning zero rewards. '
                'Set predict_sequence=True in sampler config.'
            )
            n_aa = 20
            return (
                torch.full((K,), float('-inf'), device=xl.device),
                [{} for _ in range(K)],
                xl.new_zeros(K, xl.shape[1], n_aa),
            )

        token_counts = batch["token_mask"].sum(dim=1).to(dtype=torch.int)
        if (
            torch.unique(token_counts).numel() != 1
            or int(token_counts[0].item()) != batch["token_mask"].shape[1]
        ):
            raise RuntimeError(
                "ColabFoldReward expected real-token inputs without padded tails, "
                f"but saw token counts {token_counts.tolist()} for shape "
                f"{tuple(batch['token_mask'].shape)}"
            )

        rewards = torch.full((K,), float('-inf'), device=xl.device)

        ca_mask = batch['atom_type'][:, :, 1]                                   # [K, N]
        binder_mask = 1 - batch['cond_seq_mask']                                # [K, N]
        mask = (batch['gt_atom_mask'] * ca_mask * binder_mask).bool()           # [K, N]
        ai = out.get('ai', None)
        seq_indices = ai.argmax(dim=-1) if ai is not None else None             # [K, N]

        # Read target sequence and problem info from problem JSON
        problem_name = batch.get('name', None)
        target_seq, problem_info = self._load_problem_context(problem_name)

        # Binder lengths per beam (number of Cα binder atoms = residues),
        # used to locate the cross-chain block of the PAE matrix.
        binder_lengths = [int(mask[k].sum().item()) for k in range(K)]

        template_path = None
        target_msa_filepath = (
            problem_info.get('target_msa_filepath') if problem_info is not None else None
        )
        use_fast = (
            self.handler._streamlined is not None
            and self.config.mode in ('template', 'singleseq', 'msa')
            and target_seq is not None
            and (self.config.mode != 'msa' or target_msa_filepath is not None)
        )
        if use_fast:
            if self.config.mode == 'template':
                template_path = self._get_or_create_template_dir(problem_name, problem_info)

        # Map both 3-letter and 1-letter amino acid codes to PROTEIN_RESTYPES indices.
        _AA_TO_IDX = {aa: i for i, aa in self._IDX_TO_AA.items()}
        _AA_TO_IDX.update(
            {
                aa1: _AA_TO_IDX[aa3]
                for aa1, aa3 in RESTYPE_1_TO_3.items()
                if aa3 in _AA_TO_IDX
            }
        )

        # Collect the final per-beam sequence strings (MPNN redesign or diffusion fallback)
        # before entering the TemporaryDirectory so they are available for redesigned_ai.
        aa_seqs: List[str] = []
        mpnn_total_elapsed = 0.0
        colabfold_total_elapsed = 0.0

        # tempdir is only needed for ProteinMPNN (writes PDB as input).
        # With use_proteinmpnn=False it is a zero-cost context manager.
        with tempfile.TemporaryDirectory() as tmp_dir:

            problem_name = batch.get('name', 'sample')
            pad = len(str(K - 1))

            # Build per-beam sequences (with optional ProteinMPNN redesign)
            for k in range(K):
                valid = mask[k].bool()                    # [N]
                aa_seq = (
                    ''.join(
                        self._IDX_TO_AA.get(int(idx), 'X')
                        for idx, m in zip(seq_indices[k], valid) if m
                    )
                    if seq_indices is not None
                    else ""
                )

                if self.mpnn is not None:
                    mpnn_start_time = time.perf_counter()
                    try:
                        beam_pdb_path = os.path.join(tmp_dir, f'{problem_name}_{k:0{pad}d}_mpnn_input.pdb')
                        beam_feat = {
                            key: (val[k : k + 1] if isinstance(val, torch.Tensor) else val)
                            for key, val in batch.items()
                        }
                        beam_feat["gt_atom_positions"] = xl[k : k + 1]
                        if "ai" in out:
                            beam_feat["gt_restype"] = out["ai"][k : k + 1]
                        np_list = debatchify_tensor_features(beam_feat)
                        save_np_features_to_pdb(
                            residue_index=np_list[0]["residue_index"],
                            asym_id=np_list[0]["asym_id"],
                            atom_type=np_list[0]["atom_type"],
                            atom_symbol=np_list[0]["atom_symbol"],
                            atom_mask=np_list[0]["token_struct_mask"],
                            atom_positions=np_list[0]["gt_atom_positions"],
                            restype=np_list[0].get("gt_restype"),
                            filepath=beam_pdb_path,
                            default_resname="UNK",
                        )
                        atom_valid = beam_feat["gt_atom_mask"][0].bool()
                        asym_id_k  = beam_feat["asym_id"][0][atom_valid]
                        seq_mask_k = beam_feat["cond_seq_mask"][0][atom_valid]
                        binder_asym  = torch.unique(asym_id_k[seq_mask_k == 0]).tolist()
                        target_asym  = torch.unique(asym_id_k[seq_mask_k == 1]).tolist()
                        designed_chains = [chr(ord("A") + int(a)) for a in binder_asym]
                        fixed_chains    = [chr(ord("A") + int(a)) for a in target_asym]
                        mpnn_lines = self.mpnn.predict(
                            beam_pdb_path,
                            designed_chain_list=designed_chains,
                            fixed_chain_list=fixed_chains,
                        )
                        mpnn_seqs = [
                            line for line in mpnn_lines.strip().split("\n")
                            if not line.startswith(">")
                        ]
                        if mpnn_seqs:
                            aa_seq = mpnn_seqs[0]
                            logging.info(f"[ColabFoldReward] beam {k}: ProteinMPNN redesign → {aa_seq[:20]}...")
                    except Exception as e:
                        logging.warning(f"[ColabFoldReward] ProteinMPNN failed for beam {k}: {e}")
                    finally:
                        mpnn_elapsed = time.perf_counter() - mpnn_start_time
                        mpnn_total_elapsed += mpnn_elapsed
                        logging.info(
                            f'[ColabFoldReward] ProteinMPNN runtime (beam={k}) = {mpnn_elapsed:.3f}s'
                        )

                aa_seqs.append(aa_seq)

            # ------------------------------------------------------------------
            # Fast path: in-process, no file I/O, no HHsearch rebuild
            # ------------------------------------------------------------------
            if use_fast:
                save_pdb_paths = self._make_pdb_paths(job_dirs, K)

                try:
                    colabfold_start_time = time.perf_counter()
                    raw_results = self.handler.predict_in_memory(
                        binder_seqs=aa_seqs,
                        target_seq=target_seq,
                        template_path=template_path,
                        target_msa_filepath=target_msa_filepath,
                        save_pdb_paths=save_pdb_paths,
                    )
                    colabfold_total_elapsed += time.perf_counter() - colabfold_start_time
                except Exception as e:
                    logging.warning(f'[ColabFoldReward] predict_in_memory failed: {e}')
                    return (
                        torch.full((K,), float('-inf'), device=xl.device),
                        [{} for _ in range(K)],
                        ai.clone() if ai is not None else batch["gt_restype"].clone(),
                    )

                metrics_list = []
                for k, res in enumerate(raw_results):
                    m = {
                        'i_pae': None,
                        'iptm': res['iptm'],
                        'ptm': res['ptm'],
                        'plddt': float(np.mean(res['plddt'])) if res['plddt'] is not None else 0.0,
                    }
                    pae = res.get('pae')
                    if pae is not None:
                        L = binder_lengths[k]
                        if pae.ndim == 2 and pae.shape[0] > L and pae.shape[1] > L:
                            m['i_pae'] = float(
                                (np.mean(pae[:L, L:]) + np.mean(pae[L:, :L])) / 2
                            )
                        else:
                            logging.warning(
                                f'[ColabFoldReward] beam {k}: PAE shape {pae.shape} '
                                f'unexpected for binder_length={L}'
                            )
                    else:
                        logging.warning(f'[ColabFoldReward] beam {k}: no PAE in result')

                    if m['i_pae'] is not None:
                        rewards[k] = -m['i_pae']

                    if job_dirs and k < len(job_dirs) and job_dirs[k]:
                        os.makedirs(job_dirs[k], exist_ok=True)
                        with open(os.path.join(job_dirs[k], 'colabfold_scores.json'), 'w') as f:
                            json.dump(m, f, indent=2)

                    metrics_list.append(m)

            # ------------------------------------------------------------------
            # File-based fallback (MSA mode, or _streamlined not available)
            # ------------------------------------------------------------------
            else:
                fasta_lines = []
                for k in range(K):
                    full_seq = f'{aa_seqs[k]}:{target_seq}' if target_seq else aa_seqs[k]
                    fasta_lines.append(f'>{problem_name}_{k:0{pad}d}')
                    fasta_lines.append(full_seq)
                fasta_path = os.path.join(tmp_dir, 'input.fasta')
                with open(fasta_path, 'w') as f:
                    f.write('\n'.join(fasta_lines) + '\n')

                try:
                    colabfold_start_time = time.perf_counter()
                    self.handler.fold(tmp_dir)
                    colabfold_total_elapsed += time.perf_counter() - colabfold_start_time
                except Exception as e:
                    logging.warning(f'[ColabFoldReward] ColabFold failed: {e}')
                    return (
                        torch.full((K,), float('-inf'), device=xl.device),
                        [{} for _ in range(K)],
                        ai.clone() if ai is not None else batch["gt_restype"].clone(),
                    )

                logging.info(f'[ColabFoldReward] tmp_dir contents after fold: {os.listdir(tmp_dir)}')

                metrics_list = []
                for k in range(K):
                    beam_dir = os.path.join(tmp_dir, f'{problem_name}_{k:0{pad}d}')
                    dir_contents = os.listdir(beam_dir) if os.path.isdir(beam_dir) else []
                    logging.info(f'[ColabFoldReward] beam {k}: dir={beam_dir} exists={os.path.isdir(beam_dir)} files={dir_contents}')
                    score_files = sorted(glob.glob(os.path.join(beam_dir, 'scores_*.json')))
                    m = {'i_pae': None, 'iptm': 0.0, 'ptm': 0.0, 'plddt': 0.0}

                    if score_files:
                        with open(score_files[0]) as f:
                            cf = json.load(f)
                        logging.info(f'[ColabFoldReward] beam {k}: score keys={list(cf.keys())} pae_present={"pae" in cf} pae_type={type(cf.get("pae")).__name__}')
                        m['iptm'] = float(cf.get('iptm', 0.0))
                        m['ptm']  = float(cf.get('ptm', 0.0))
                        plddt = cf.get('plddt', [])
                        m['plddt'] = float(np.mean(plddt)) if plddt else 0.0

                        if 'pae' in cf:
                            pae = np.array(cf['pae'])
                            L = binder_lengths[k]
                            if pae.ndim == 2 and pae.shape[0] > L and pae.shape[1] > L:
                                m['i_pae'] = float(
                                    (np.mean(pae[:L, L:]) + np.mean(pae[L:, :L])) / 2
                                )
                            else:
                                logging.warning(
                                    f'[ColabFoldReward] beam {k}: PAE shape {pae.shape} '
                                    f'unexpected for binder_length={L}'
                                )
                        else:
                            logging.warning(
                                f'[ColabFoldReward] beam {k}: no "pae" key in {score_files[0]}'
                            )

                        if m['i_pae'] is not None:
                            rewards[k] = -m['i_pae']

                        if job_dirs and k < len(job_dirs) and job_dirs[k]:
                            os.makedirs(job_dirs[k], exist_ok=True)
                            for pdb in glob.glob(os.path.join(beam_dir, '*.pdb')):
                                shutil.copy2(pdb, job_dirs[k])
                            with open(os.path.join(job_dirs[k], 'colabfold_scores.json'), 'w') as f:
                                json.dump(m, f, indent=2)

                    metrics_list.append(m)

        logging.info(
            '[ColabFoldReward] checkpoint runtime breakdown '
            f'(problem={problem_name}, beams={K}) '
            f'proteinmpnn={mpnn_total_elapsed:.3f}s '
            f'colabfold={colabfold_total_elapsed:.3f}s'
        )

        # Build redesigned_ai: start from the diffusion sequence when present,
        # otherwise from the conditioned batch restypes, then replace each
        # binder residue with the final scoring sequence on every token that
        # belongs to that residue (not just the C-alpha token).
        redesigned_ai = ai.clone() if ai is not None else batch["gt_restype"].clone()
        for k in range(K):
            ca_positions = mask[k].nonzero(as_tuple=True)[0].tolist()
            for pos_in_seq, ca_pos in enumerate(ca_positions):
                if pos_in_seq >= len(aa_seqs[k]):
                    break

                aa_char = aa_seqs[k][pos_in_seq]
                residue_index = batch["residue_index"][k, ca_pos]
                asym_id = batch["asym_id"][k, ca_pos]
                residue_token_mask = (
                    (batch["residue_index"][k] == residue_index)
                    & (batch["asym_id"][k] == asym_id)
                )

                fallback_idx = (
                    int(seq_indices[k, ca_pos].item()) if seq_indices is not None else 0
                )
                aa_idx = _AA_TO_IDX.get(aa_char, fallback_idx)
                redesigned_ai[k, residue_token_mask, :] = 0
                redesigned_ai[k, residue_token_mask, aa_idx] = 1

        return rewards, metrics_list, redesigned_ai
