"""
Beam search for inference-time scaling over diffusion trajectories.

At each scoring checkpoint, the current N beams are branched into N² candidates
(each beam produces N copies with optional small noise). All N² candidates are
fully denoised to t=0 (look-ahead), scored by a configurable reward function,
and the top N are kept to continue. All N² evaluated structures from every
checkpoint are collected in a global pool; the top `n_output` by score are
returned after the run.

Output layout (example: N=4, 2 checkpoints):

    {outdir}/{problem}/
      search/
        job0/          ← checkpoint 0, 16 candidates evaluated
          t0b0/        ← beam 0, branch copy 0
            denoised.pdb
            rank_001_model_1.pdb   ← ColabFold predicted structure(s)
            colabfold_scores.json
          t0b1/ ... t3b3/
        job1/          ← top-4 survivors from job0, each branched again
          t0b2b0/ ...  ← name encodes full lineage
        rewards.csv    ← all 32 candidate rows
      pdbs/
        {problem}_0.pdb   ← rank 0 (best overall)
        ...
      index.csv           ← rank → job, dir, reward, i_pae, iptm, ptm, plddt
"""

import os
import csv
import json
import time
import logging
import shutil
import torch
from typing import Dict, List, Optional, TYPE_CHECKING

from genie3.generation.diffusion.reward.base import Reward
from genie3.generation.utils.feat_utils import (
    crop_tensor_features_to_token_mask,
    debatchify_tensor_features,
    pad_tensor_features_to_n_token,
    save_np_features_to_pdb,
)

if TYPE_CHECKING:
    from genie3.generation.diffusion.sampler.ddim import DDIMSampler


class BeamSearch:
    """
    Beam search orchestrator for diffusion sampling.

    At each scoring checkpoint the current N beams are branched into N²
    candidates (repeat_interleave + optional noise). All N² are fully
    denoised (look-ahead), scored, and the top N continue. All evaluated
    structures across all checkpoints are pooled; top `n_output` are returned.
    """

    def __init__(
        self,
        beam_width: int,
        score_interval: int,
        reward: Reward,
        n_output: int = -1,
        branch_noise_scale: float = 0.0,
        outdir: Optional[str] = None,
        requested_n_sample: Optional[int] = None,
    ):
        """
        Args:
            beam_width: Number of parallel trajectories (N). At each checkpoint
                        these branch into N² candidates; top N are kept.
            score_interval: Denoising steps between reward evaluations.
            reward: Instantiated Reward object used to score beams.
            n_output: Structures to return from the global pool. Defaults to
                      beam_width when set to -1.
            branch_noise_scale: Standard deviation of noise added to each beam
                copy when branching. 0 = deterministic copies (diversity comes
                from the look-ahead denoising).
            outdir: Root output directory (e.g. config.io.outdir). Structured
                    subdirs are created automatically inside outdir/{problem}/.
            requested_n_sample: User-facing requested number of final outputs
                for this problem. Used to trim the last beam-search run.
        """
        self.beam_width = beam_width
        self.score_interval = score_interval
        self.reward = reward
        self.n_output = n_output if n_output > 0 else beam_width
        self.branch_noise_scale = branch_noise_scale
        self.outdir = outdir
        self.requested_n_sample = requested_n_sample
        self._fixed_binder_max_length: Optional[int] = None

    def set_fixed_token_padding(self, binder_max_length: Optional[int]) -> None:
        """Enable fixed-shape denoiser padding for a single binder-design problem."""
        self._fixed_binder_max_length = binder_max_length

    def _get_fixed_n_token(self, batch: Dict[str, torch.Tensor]) -> Optional[int]:
        """
        Compute the fixed padded token length for the current batch.

        Real tokens remain a prefix of the padded batch; only binder tokens are
        variable across samples of the same problem, so the required padding is
        the gap between the sampled binder length and binder_max_length.
        """
        if self._fixed_binder_max_length is None:
            return None

        binder_lengths = (
            ((1 - batch["cond_seq_mask"]) * batch["gt_atom_mask"])
            .sum(dim=1)
            .to(dtype=torch.int)
        )
        if torch.unique(binder_lengths).numel() != 1:
            raise RuntimeError(
                "Fixed-shape beam-search padding expects a single binder length per batch, "
                f"but found {binder_lengths.tolist()}"
            )

        binder_length = int(binder_lengths[0].item())
        if binder_length > self._fixed_binder_max_length:
            raise RuntimeError(
                "Observed binder length exceeds configured binder_max_length: "
                f"{binder_length} > {self._fixed_binder_max_length}"
            )

        return batch["token_mask"].shape[1] + (
            self._fixed_binder_max_length - binder_length
        )

    def _maybe_pad_batch_for_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Pad denoiser-facing tensors to a fixed per-problem shape when configured."""
        fixed_n_token = self._get_fixed_n_token(batch)
        if fixed_n_token is None:
            return batch
        return pad_tensor_features_to_n_token(batch, fixed_n_token)

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _tile_batch(self, batch: Dict[str, torch.Tensor], k: int) -> Dict[str, torch.Tensor]:
        """Tile all tensor values in batch k times along the batch dimension."""
        tiled = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                tiled[key] = val.repeat_interleave(k, dim=0)
            else:
                tiled[key] = val
        return tiled

    def _branch_beams(
        self,
        xl: torch.Tensor,
        beam_batch: Dict[str, torch.Tensor],
    ):
        """
        Expand K beams → K² candidates via repeat_interleave + optional noise.

        Args:
            xl: Current beam coordinates. Shape: [K, N, 3]
            beam_batch: Tiled conditioning batch for K beams.

        Returns:
            (xl_exp [K², N, 3], batch_exp) — K² branched candidates.
        """
        K = xl.shape[0]
        xl_exp = xl.repeat_interleave(K, dim=0)        # [K², N, 3]
        if self.branch_noise_scale > 0:
            xl_exp = xl_exp + torch.randn_like(xl_exp) * self.branch_noise_scale
        batch_exp = {
            key: (val.repeat_interleave(K, dim=0) if isinstance(val, torch.Tensor) else val)
            for key, val in beam_batch.items()
        }
        return xl_exp, batch_exp

    def _step_in_chunks(
        self,
        sampler: "DDIMSampler",
        model,
        batch: Dict[str, torch.Tensor],
        xs: torch.Tensor,
        s: torch.Tensor,
        step_size: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Run sampler._step over the batch in smaller chunks."""
        if xs.shape[0] <= chunk_size:
            return sampler._step(model=model, batch=batch, xs=xs, s=s, step_size=step_size)

        chunks = []
        for start in range(0, xs.shape[0], chunk_size):
            end = min(start + chunk_size, xs.shape[0])
            batch_chunk = {
                key: (val[start:end] if isinstance(val, torch.Tensor) else val)
                for key, val in batch.items()
            }
            chunks.append(
                sampler._step(
                    model=model,
                    batch=batch_chunk,
                    xs=xs[start:end],
                    s=s[start:end],
                    step_size=step_size,
                )
            )
        return torch.cat(chunks, dim=0)

    def _sample_sequence_in_chunks(
        self,
        sampler: "DDIMSampler",
        model,
        batch: Dict[str, torch.Tensor],
        xl: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """Run sampler._sample_sequence over the batch in smaller chunks."""
        if xl.shape[0] <= chunk_size:
            return sampler._sample_sequence(model, batch, xl)

        chunks = []
        for start in range(0, xl.shape[0], chunk_size):
            end = min(start + chunk_size, xl.shape[0])
            batch_chunk = {
                key: (val[start:end].clone() if isinstance(val, torch.Tensor) else val)
                for key, val in batch.items()
            }
            chunks.append(
                sampler._sample_sequence(
                    model,
                    batch_chunk,
                    xl[start:end],
                )
            )
        return torch.cat(chunks, dim=0)

    def _save_denoised_pdb(
        self,
        xl_beam: torch.Tensor,
        ai_beam: Optional[torch.Tensor],
        beam_batch_single: Dict,
        filepath: str,
    ):
        """
        Save a single fully-denoised beam as a PDB file.

        Args:
            xl_beam: Coordinates, shape [1, N, 3].
            ai_beam: Sequence one-hots, shape [1, N, n_aa], or None.
            beam_batch_single: Batch dict sliced to one beam.
            filepath: Destination PDB path.
        """
        feat = dict(beam_batch_single)
        feat['gt_atom_positions'] = xl_beam
        if ai_beam is not None:
            feat['gt_restype'] = ai_beam
        np_list = debatchify_tensor_features(feat)
        save_np_features_to_pdb(
            residue_index=np_list[0]['residue_index'],
            asym_id=np_list[0]['asym_id'],
            atom_type=np_list[0]['atom_type'],
            atom_symbol=np_list[0]['atom_symbol'],
            atom_mask=np_list[0]['token_struct_mask'],
            atom_positions=np_list[0]['gt_atom_positions'],
            restype=np_list[0].get('gt_restype'),
            filepath=filepath,
            default_resname='UNK',
        )

    def _should_skip_sequence_sampling(self) -> bool:
        """
        Sequence decoding is unnecessary when the reward always redesigns with
        ProteinMPNN before scoring.
        """
        return getattr(self.reward, "mpnn", None) is not None

    # ------------------------------------------------------------------
    # Main beam search loop
    # ------------------------------------------------------------------

    def run(
        self,
        sampler: "DDIMSampler",
        model,
        batch: Dict[str, torch.Tensor],
        n_timestep: int,
        n_sample_step: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Run beam search over the denoising trajectory.

        At each scoring checkpoint:
        1. Branch N beams → N² candidates (with optional noise).
        2. Look-ahead: fully denoise all N² to t=0.
        3. Sample sequence on clean structures.
        4. Score all N² (ColabFold / reward).
        5. Keep top N to continue; accumulate all N² in global pool.

        After all steps, return top `n_output` from the global pool.

        Args:
            sampler: DDIMSampler instance.
            model: Denoiser model.
            batch: Conditioning batch (original batch size B=1).
            n_timestep: Total diffusion timesteps.
            n_sample_step: Number of sampling steps.

        Returns:
            dict with:
                'xl'     — top-n_output coordinates [n_output, N, 3]
                'ai'     — top-n_output sequence one-hots [n_output, N, n_aa]
                'scores' — corresponding reward scores [n_output]
        """
        device = batch["gt_atom_positions"].device
        problem_name = batch.get('name', 'sample')
        # Use the full sample name for output paths so that multiple samples of
        # the same problem (processed on different devices) write to distinct dirs.
        sample_name = batch.get('sample_name', problem_name)

        batch = self._maybe_pad_batch_for_model(batch)

        # Expand batch and initial noise to beam_width
        beam_batch = self._tile_batch(batch, self.beam_width)
        xl = torch.randn_like(beam_batch["gt_atom_positions"])   # [K, N, 3]

        step_size = n_timestep // n_sample_step
        steps = list(reversed(range(1, n_timestep + 1, step_size)))
        batch_size = xl.shape[0]

        # Beam lineage labels: each surviving beam carries its history string
        # Initial: ['t0', 't1', ..., 't{K-1}']
        K = self.beam_width
        beam_labels: List[str] = [f't{i}' for i in range(K)]

        # Global pool: all evaluated candidates from every checkpoint
        global_pool: List[Dict] = []
        checkpoint_idx = 0

        for i, step in enumerate(steps):
            checkpoint_start_time = time.perf_counter()
            t = torch.full((batch_size,), step, dtype=torch.int, device=device)
            xl = sampler._step(model=model, batch=beam_batch, xs=xl, s=t, step_size=step_size)

            if xl is None:
                return None

            if (i + 1) % self.score_interval == 0:

                # 1. Branch: K beams → K² candidates
                branch_start_time = time.perf_counter()
                xl_branched, batch_branched = self._branch_beams(xl, beam_batch)
                K2 = xl_branched.shape[0]   # == beam_width²
                branch_elapsed = time.perf_counter() - branch_start_time

                # Assign lineage labels: parent p = b // K, copy c = b % K
                branched_labels = [
                    f'{beam_labels[b // K]}b{b % K}' for b in range(K2)
                ]

                # 2. Look-ahead: fully denoise K² candidates to t=0
                lookahead_start_time = time.perf_counter()
                xl_clean = xl_branched
                for future_step in steps[i + 1:]:
                    t_clean = torch.full((K2,), future_step, dtype=torch.int, device=device)
                    xl_clean = self._step_in_chunks(
                        sampler=sampler,
                        model=model, batch=batch_branched,
                        xs=xl_clean, s=t_clean, step_size=step_size,
                        chunk_size=K,
                    )
                lookahead_elapsed = time.perf_counter() - lookahead_start_time

                # 3. Sample sequence on K² clean structures
                sequence_start_time = time.perf_counter()
                sampled_ai = None
                if not self._should_skip_sequence_sampling():
                    score_batch = dict(batch_branched)
                    for mutable_key in ("gt_restype", "cond_seq_mask"):
                        if mutable_key in score_batch and isinstance(score_batch[mutable_key], torch.Tensor):
                            score_batch[mutable_key] = score_batch[mutable_key].clone()
                    with torch.inference_mode():
                        sampled_ai = self._sample_sequence_in_chunks(
                            sampler,
                            model,
                            score_batch,
                            xl_clean,
                            chunk_size=K,
                        )
                score_out = {'ai': sampled_ai} if sampled_ai is not None else {}
                sequence_elapsed = time.perf_counter() - sequence_start_time

                # 4a. Create output dirs (before reward.compute — dirs passed as job_dirs)
                io_start_time = time.perf_counter()
                beam_dirs: Optional[List[str]] = None
                if self.outdir is not None:
                    beam_dirs = []
                    for b in range(K2):
                        bdir = os.path.join(
                            self.outdir, problem_name, 'jobs', sample_name,
                            f'job{checkpoint_idx}', branched_labels[b],
                        )
                        os.makedirs(bdir, exist_ok=True)
                        beam_dirs.append(bdir)
                io_dir_elapsed = time.perf_counter() - io_start_time

                # 5. Score K² beams (ColabFold → -i_pae reward) — assigns redesigned_ai
                reward_batch = crop_tensor_features_to_token_mask(batch_branched)
                real_n_token = reward_batch["token_mask"].shape[1]
                reward_xl = xl_clean[:, :real_n_token]
                reward_out = (
                    {"ai": sampled_ai[:, :real_n_token]}
                    if sampled_ai is not None
                    else {}
                )
                reward_start_time = time.perf_counter()
                scores, metrics, redesigned_ai = self.reward.compute(
                    batch=reward_batch, xl=reward_xl, out=reward_out, job_dirs=beam_dirs
                )
                reward_elapsed = time.perf_counter() - reward_start_time

                # 4b. Save denoised PDBs (after reward.compute — redesigned_ai now available)
                io_save_start_time = time.perf_counter()
                if self.outdir is not None:
                    for b in range(K2):
                        beam_feat_b = {
                            key: (val[b:b+1] if isinstance(val, torch.Tensor) else val)
                            for key, val in reward_batch.items()
                        }
                        ai_b = redesigned_ai[b:b+1] if redesigned_ai is not None else (
                            reward_out["ai"][b:b+1] if "ai" in reward_out else None
                        )
                        self._save_denoised_pdb(
                            reward_xl[b:b+1], ai_b, beam_feat_b,
                            os.path.join(beam_dirs[b], 'denoised.pdb'),
                        )
                io_save_elapsed = time.perf_counter() - io_save_start_time

                select_start_time = time.perf_counter()
                logging.info(
                    f"[BeamSearch] checkpoint={checkpoint_idx} step={step} "
                    f"top_score={scores.max():.4f}"
                )

                # 6. Accumulate in global pool
                global_pool.append({
                    'scores':         scores.detach(),
                    'xl_clean':       reward_xl.detach(),
                    'ai':             redesigned_ai.detach() if redesigned_ai is not None else (
                                          reward_out['ai'].detach() if reward_out['ai'] is not None else None
                                      ),
                    'beam_dirs':      beam_dirs or [],
                    'beam_labels':    branched_labels,
                    'metrics':        metrics,
                    'checkpoint_idx': checkpoint_idx,
                })

                # 7. Down-select top beam_width from K² to continue
                top_k_idx = torch.topk(scores, K).indices   # [K]
                xl = xl_branched[top_k_idx]                  # [K, N, 3]
                beam_batch = {
                    key: (val[top_k_idx] if isinstance(val, torch.Tensor) else val)
                    for key, val in batch_branched.items()
                }
                beam_labels = [branched_labels[idx] for idx in top_k_idx.tolist()]
                batch_size = xl.shape[0]
                select_elapsed = time.perf_counter() - select_start_time
                checkpoint_elapsed = time.perf_counter() - checkpoint_start_time
                logging.info(
                    '[BeamSearch] checkpoint runtime breakdown '
                    f'(checkpoint={checkpoint_idx}, step={step}) '
                    f'branch={branch_elapsed:.3f}s '
                    f'lookahead={lookahead_elapsed:.3f}s '
                    f'sequence={sequence_elapsed:.3f}s '
                    f'mkdir={io_dir_elapsed:.3f}s '
                    f'reward={reward_elapsed:.3f}s '
                    f'save_pdb={io_save_elapsed:.3f}s '
                    f'select={select_elapsed:.3f}s '
                    f'total={checkpoint_elapsed:.3f}s'
                )
                checkpoint_idx += 1

        # ------------------------------------------------------------------
        # Post-loop: write rewards.csv, select top n_output, write index.csv
        # ------------------------------------------------------------------

        if not global_pool:
            logging.warning(
                "[BeamSearch] No scoring checkpoint fired "
                f"(score_interval={self.score_interval} > n_sample_step={n_sample_step}). "
                "Returning first beam without reward ranking."
            )
            return {"xl": xl[0:1]}

        # Flat list of all evaluated candidates across all checkpoints
        flat = []
        for entry in global_pool:
            dirs = entry.get('beam_dirs', [])
            labels = entry['beam_labels']
            for b, (bdir, label, sc, m) in enumerate(zip(
                    dirs, labels,
                    entry['scores'].tolist(),
                    entry['metrics'])):
                flat.append({
                    'job':    entry['checkpoint_idx'],
                    'dir':    label,
                    'bdir':   bdir,
                    'reward': sc,
                    **{k: m.get(k, '') for k in ['i_pae', 'iptm', 'ptm', 'plddt']},
                })

        # Merge tensors for top-n selection
        all_scores = torch.cat([e['scores']   for e in global_pool])
        all_xl     = torch.cat([e['xl_clean'] for e in global_pool])
        all_ai_list = [e['ai'] for e in global_pool if e['ai'] is not None]
        all_ai = torch.cat(all_ai_list) if all_ai_list else None

        requested_outputs_this_run = self.n_output
        if self.requested_n_sample is not None:
            try:
                sample_idx = int(sample_name.rsplit('_', 1)[-1])
            except ValueError:
                sample_idx = 0
            remaining_outputs = self.requested_n_sample - sample_idx * self.n_output
            requested_outputs_this_run = max(0, min(self.n_output, remaining_outputs))

        n_select = min(requested_outputs_this_run, all_scores.shape[0])
        top_idx  = torch.topk(all_scores, n_select).indices

        # Write per-sample output and contribute to shared pdbs/
        if self.outdir is not None:
            job_dir = os.path.join(self.outdir, problem_name, 'jobs', sample_name)
            os.makedirs(job_dir, exist_ok=True)

            # Per-sample rewards.csv — full candidate history for this beam search run
            csv_path = os.path.join(job_dir, 'rewards.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['job', 'dir', 'reward', 'i_pae', 'iptm', 'ptm', 'plddt'])
                writer.writeheader()
                for row in flat:
                    writer.writerow({k: row[k] for k in writer.fieldnames})

            # Copy top-K denoised PDBs to shared pdbs/ with globally unique indices.
            # global_index = sample_idx * n_output + beam_rank so that each sample
            # occupies a contiguous, non-overlapping range and devices never collide.
            pdbs_dir = os.path.join(self.outdir, problem_name, 'pdbs')
            os.makedirs(pdbs_dir, exist_ok=True)
            try:
                sample_idx = int(sample_name.rsplit('_', 1)[-1])
            except ValueError:
                sample_idx = 0
            summary_rows = []
            for beam_rank, idx in enumerate(top_idx.tolist()):
                if idx < len(flat):
                    row = flat[idx]
                    src = os.path.join(row['bdir'], 'denoised.pdb') if row['bdir'] else None
                    global_idx = sample_idx * self.n_output + beam_rank
                    if src and os.path.exists(src):
                        shutil.copy2(src, os.path.join(pdbs_dir, f'{problem_name}_{global_idx}.pdb'))
                    summary_rows.append({
                        'index':  global_idx,
                        'sample': sample_name,
                        'rank':   beam_rank,
                        **{k: row.get(k, '') for k in ['reward', 'i_pae', 'iptm', 'ptm', 'plddt']},
                    })

            # summary.json — read by rank 0 in on_test_end to build aggregate rewards.csv
            summary_path = os.path.join(job_dir, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_rows, f, indent=2)

        result = {
            "xl":     all_xl[top_idx],
            "scores": all_scores[top_idx],
        }
        if all_ai is not None:
            result["ai"] = all_ai[top_idx]
        return result
