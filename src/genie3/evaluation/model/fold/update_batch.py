"""
Streamlined in-process ColabFold for beam search scoring.

Bypasses colabfold.batch.run() to eliminate repeated overhead on every
scoring checkpoint:
  - HHsearch DB rebuild (mk_hhsearch_db) on every call
  - Target chain feature recomputation (generate_input_feature) on every call
  - matplotlib visualization I/O (coverage, pAE, pLDDT plots)
  - Bookkeeping file I/O (config.json, bibtex, .a3m, .done.txt)
  - Scores JSON round-trip (write then immediately re-parse)

Target chain features are computed once via prepare(), cached, and reused
across all beams and all scoring checkpoints for the same problem.
"""

import copy
import logging
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from genie3.evaluation.utils.msa import build_complex_msa_text

logger = logging.getLogger(__name__)


def _multimer_ranking_score(result: Dict[str, Any]) -> float:
    """
    Match ColabFold multimer ranking as closely as possible.

    ColabFold ranks multimer predictions using a weighted ipTM/pTM score.
    """
    return (0.8 * float(result.get('iptm', 0.0))) + (0.2 * float(result.get('ptm', 0.0)))


class StreamlinedColabFold:
    """
    Stripped-down ColabFold inference engine for beam search scoring.

    Calls model_runner.predict() directly — no subprocess, no file I/O,
    no hhsearch, no matplotlib.

    Target chain features are built once by prepare() and reused for every
    beam in every scoring checkpoint.  JAX/XLA compilation is triggered
    lazily on the first predict() call for a given total sequence length;
    subsequent calls with the same length reuse the compiled trace.
    """

    def __init__(
        self,
        model_runner_and_params: List[Tuple],
        model_type: str = 'alphafold2_multimer_v3',
        mode: str = 'template',
        use_templates: bool = True,
        max_seq: int = 5,
        max_extra_seq: int = 1,
    ):
        """
        Args:
            model_runner_and_params: List of (model_name, model_runner, params)
                tuples as returned by load_models_and_params().
            model_type: AF2 model type string (must contain 'multimer').
            use_templates: Whether to use structural templates for the target.
            max_seq: max_seq passed to process_multimer_features.
            max_extra_seq: max_extra_seq (unused directly, kept for parity).
        """
        self.model_runner_and_params = model_runner_and_params
        self.model_type = model_type
        self.mode = mode
        self.use_templates = use_templates
        self.max_seq = max_seq
        self.max_extra_seq = max_extra_seq

        self._cached_target_seq: Optional[str] = None
        self._cached_template_path: Optional[str] = None
        self._cached_target_msa_filepath: Optional[str] = None
        self._cached_target_chain_feats: Optional[List[Dict]] = None  # one dict per target chain

        # Tracks the last total_len for which JAX has been compiled.
        self._warmed_up_for: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        target_seq: str,
        template_path: Optional[str] = None,
        target_msa_filepath: Optional[str] = None,
    ):
        """
        Build and cache per-chain features for the target.  Idempotent —
        skips work when called again with the same target_seq/template_path.

        mk_hhsearch_db must have already been called on template_path (done
        once by ColabFoldHandler._get_or_create_template_dir).

        Args:
            target_seq: Target protein amino acid sequence.
            template_path: Directory containing a pre-built pdb70 DB.
                           Pass None to use mock (zero-fill) templates.
        """
        if (
            target_seq == self._cached_target_seq
            and template_path == self._cached_template_path
            and target_msa_filepath == self._cached_target_msa_filepath
        ):
            return  # already cached for this problem

        self._cached_target_seq = target_seq
        self._cached_template_path = template_path
        self._cached_target_msa_filepath = target_msa_filepath

        if self.mode == 'msa':
            logger.debug(
                f'[StreamlinedColabFold] Caching target MSA context '
                f'(len={len(target_seq)}, msa={target_msa_filepath is not None})'
            )
            self._cached_target_chain_feats = None
            return

        from colabfold.batch import (
            mk_mock_template,
            mk_template,
            build_monomer_feature,
            build_multimer_feature,
        )

        target_chains = target_seq.split(':')
        total_len = sum(len(c) for c in target_chains)
        logger.debug(
            f'[StreamlinedColabFold] Building target chain features '
            f'(n_chains={len(target_chains)}, total_len={total_len}, '
            f'templates={template_path is not None})'
        )

        per_chain_feats = []
        for chain_seq in target_chains:
            chain_a3m = f'>101\n{chain_seq}\n'
            if template_path is not None and self.use_templates:
                template_feat = mk_template(chain_a3m, template_path, chain_seq)
            else:
                template_feat = mk_mock_template(chain_seq)
            per_chain_feats.append({
                **build_monomer_feature(chain_seq, chain_a3m, template_feat),
                **build_multimer_feature(chain_a3m),
            })

        self._cached_target_chain_feats = per_chain_feats

    def predict(
        self,
        binder_seq: str,
        random_seed: int = 0,
        save_pdb_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run one AF2 multimer forward pass for binder_seq + cached target.

        No file I/O unless save_pdb_path is specified.  Must call prepare()
        first.

        Args:
            binder_seq: Binder amino acid sequence.
            random_seed: Passed to model_runner.predict().
            save_pdb_path: Optional file path for the unrelaxed PDB output.

        Returns:
            dict with keys:
                pae   — np.ndarray [L, L] (L = binder + target length)
                plddt — np.ndarray [L]
                ptm   — float
                iptm  — float
        """
        results = self.predict_all(
            binder_seq=binder_seq,
            random_seed=random_seed,
            save_pdb_path=save_pdb_path,
        )
        top_result = dict(results[0])
        top_result.pop('model_result', None)
        top_result.pop('feature_dict', None)
        return top_result

    def predict_all(
        self,
        binder_seq: str,
        random_seed: int = 0,
        save_pdb_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all loaded AF2 models and return them in ColabFold rank order.

        When save_pdb_path is provided, only the top-ranked prediction is
        written there. Callers that need all model outputs should save them
        explicitly from the returned results.
        """
        if self.mode == 'msa':
            return self._predict_msa(
                binder_seq=binder_seq,
                random_seed=random_seed,
                save_pdb_path=save_pdb_path,
            )
        return self._predict_template_or_singleseq(
            binder_seq=binder_seq,
            random_seed=random_seed,
            save_pdb_path=save_pdb_path,
        )

    def _predict_template_or_singleseq(
        self,
        binder_seq: str,
        random_seed: int = 0,
        save_pdb_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from colabfold.batch import (
            mk_mock_template,
            build_monomer_feature,
            build_multimer_feature,
            process_multimer_features,
        )
        from alphafold.common import protein as prot_module

        if self._cached_target_chain_feats is None:
            raise RuntimeError(
                '[StreamlinedColabFold] Call prepare() before predict()'
            )

        binder_feature_start_time = time.perf_counter()
        binder_a3m = f'>101\n{binder_seq}\n'
        binder_chain_feat = {
            **build_monomer_feature(
                binder_seq, binder_a3m, mk_mock_template(binder_seq)
            ),
            **build_multimer_feature(binder_a3m),
        }
        binder_feature_elapsed = time.perf_counter() - binder_feature_start_time

        # Deep-copy the cached target features: process_multimer_features /
        # process_unmerged_features may mutate chain dicts in-place, which
        # would corrupt the cache on the second and subsequent predict() calls.
        merge_start_time = time.perf_counter()
        features_for_chain = {prot_module.PDB_CHAIN_IDS[0]: binder_chain_feat}
        for i, chain_feat in enumerate(self._cached_target_chain_feats):
            features_for_chain[prot_module.PDB_CHAIN_IDS[i + 1]] = copy.deepcopy(chain_feat)
        feature_dict = process_multimer_features(
            features_for_chain, min_num_seq=self.max_seq + 4
        )
        # Normalise asym_id: mirrors predict_structure() in batch.py line 373
        feature_dict['asym_id'] = (
            feature_dict['asym_id'] - feature_dict['asym_id'][..., 0]
        )
        merge_elapsed = time.perf_counter() - merge_start_time

        predict_start_time = time.perf_counter()
        ranked_results = []
        for model_name, model_runner, params in self.model_runner_and_params:
            model_runner.params = params
            result, _recycles = model_runner.predict(
                feature_dict, random_seed=random_seed
            )
            target_total_len = sum(len(c) for c in self._cached_target_seq.split(':'))
            seq_len = len(binder_seq) + target_total_len
            plddt = np.array(result['plddt'][:seq_len], dtype=np.float32)
            pae = result.get('predicted_aligned_error')
            if pae is not None:
                pae = np.array(pae[:seq_len, :seq_len], dtype=np.float32)

            ranked_results.append({
                'pae': pae,
                'plddt': plddt,
                'ptm': float(result.get('ptm', 0.0)),
                'iptm': float(result.get('iptm', 0.0)),
                'model_name': model_name,
                'model_result': result,
                'feature_dict': feature_dict,
                'ranking_score': _multimer_ranking_score(result),
            })
        ranked_results.sort(key=lambda x: x['ranking_score'], reverse=True)
        for rank_idx, res in enumerate(ranked_results, start=1):
            res['rank'] = rank_idx
        predict_elapsed = time.perf_counter() - predict_start_time
        target_total_len = sum(len(c) for c in self._cached_target_seq.split(':'))
        logger.debug(
            '[StreamlinedColabFold] predict runtime '
            f'(binder_len={len(binder_seq)}, target_len={target_total_len}, '
            f'random_seed={random_seed}, save_pdb={save_pdb_path is not None}) = '
            f'{predict_elapsed:.3f}s '
            f'[binder_feat={binder_feature_elapsed:.3f}s, '
            f'merge={merge_elapsed:.3f}s]'
        )

        if save_pdb_path is not None:
            self._write_prediction_pdb(
                feature_dict=feature_dict,
                result=ranked_results[0]['model_result'],
                save_pdb_path=save_pdb_path,
                binder_seq=binder_seq,
                target_seq=self._cached_target_seq,
                remove_leading_feature_dimension=False,
            )

        return ranked_results

    def _predict_msa(
        self,
        binder_seq: str,
        random_seed: int = 0,
        save_pdb_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from colabfold.batch import unserialize_msa, generate_input_feature

        if self._cached_target_seq is None or self._cached_target_msa_filepath is None:
            raise RuntimeError(
                '[StreamlinedColabFold] Call prepare(..., target_msa_filepath=...) '
                'before MSA predict()'
            )

        binder_feature_start_time = time.perf_counter()
        complex_a3m = build_complex_msa_text(
            target_msa_filepath=self._cached_target_msa_filepath,
            binder_sequence=binder_seq,
        )
        (
            unpaired_msa,
            paired_msa,
            query_seqs_unique,
            query_seqs_cardinality,
            template_features,
        ) = unserialize_msa([complex_a3m], f'{binder_seq}:{self._cached_target_seq}')
        feature_dict, _domain_names = generate_input_feature(
            query_seqs_unique=query_seqs_unique,
            query_seqs_cardinality=query_seqs_cardinality,
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa,
            template_features=template_features,
            is_complex=True,
            model_type=self.model_type,
            max_seq=self.max_seq,
        )
        feature_dict['asym_id'] = feature_dict['asym_id'] - feature_dict['asym_id'][..., 0]
        binder_feature_elapsed = time.perf_counter() - binder_feature_start_time

        predict_start_time = time.perf_counter()
        ranked_results = []
        for model_name, model_runner, params in self.model_runner_and_params:
            model_runner.params = params
            result, _recycles = model_runner.predict(
                feature_dict, random_seed=random_seed
            )
            seq_len = len(binder_seq) + len(self._cached_target_seq)
            plddt = np.array(result['plddt'][:seq_len], dtype=np.float32)
            pae = result.get('predicted_aligned_error')
            if pae is not None:
                pae = np.array(pae[:seq_len, :seq_len], dtype=np.float32)

            ranked_results.append({
                'pae': pae,
                'plddt': plddt,
                'ptm': float(result.get('ptm', 0.0)),
                'iptm': float(result.get('iptm', 0.0)),
                'model_name': model_name,
                'model_result': result,
                'feature_dict': feature_dict,
                'ranking_score': _multimer_ranking_score(result),
            })
        ranked_results.sort(key=lambda x: x['ranking_score'], reverse=True)
        for rank_idx, res in enumerate(ranked_results, start=1):
            res['rank'] = rank_idx
        predict_elapsed = time.perf_counter() - predict_start_time
        logger.debug(
            '[StreamlinedColabFold] predict runtime '
            f'(binder_len={len(binder_seq)}, target_len={len(self._cached_target_seq)}, '
            f'random_seed={random_seed}, save_pdb={save_pdb_path is not None}) = '
            f'{predict_elapsed:.3f}s '
            f'[binder_feat={binder_feature_elapsed:.3f}s, merge={0.0:.3f}s]'
        )

        if save_pdb_path is not None:
            self._write_prediction_pdb(
                feature_dict=feature_dict,
                result=ranked_results[0]['model_result'],
                save_pdb_path=save_pdb_path,
                binder_seq=binder_seq,
                target_seq=self._cached_target_seq,
                remove_leading_feature_dimension=False,
            )

        return ranked_results

    def _write_prediction_pdb(
        self,
        feature_dict: Dict[str, Any],
        result: Dict[str, Any],
        save_pdb_path: str,
        binder_seq: str,
        target_seq: str,
        remove_leading_feature_dimension: bool = False,
    ) -> None:
        from alphafold.common import protein as prot_module

        pdb_start_time = time.perf_counter()
        Path(save_pdb_path).parent.mkdir(parents=True, exist_ok=True)
        final_atom_mask = result['structure_module']['final_atom_mask']
        b_factors = result['plddt'][:, None] * final_atom_mask
        unrelaxed_protein = prot_module.from_prediction(
            features=feature_dict,
            result=result,
            b_factors=b_factors,
            remove_leading_feature_dimension=remove_leading_feature_dimension,
        )
        Path(save_pdb_path).write_text(prot_module.to_pdb(unrelaxed_protein))
        pdb_elapsed = time.perf_counter() - pdb_start_time
        logger.debug(
            '[StreamlinedColabFold] pdb write runtime '
            f'(binder_len={len(binder_seq)}, target_len={len(target_seq)}) = '
            f'{pdb_elapsed:.3f}s'
        )
