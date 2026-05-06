"""Evaluation summary helpers for the Genie3 CLI."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re


AF_PREDICTION_FILE_RE = re.compile(
    r"^(?:unrelaxed|relaxed)_rank_\d{3}_alphafold2_(?:ptm|multimer_v3)_model_\d_seed_\d{3}\.(?:pdb|cif)$"
)


@dataclass(frozen=True)
class SuccessSummary:
    label: str
    successes: int
    unique_tm_0_5: int
    unique_tm_0_6: int


@dataclass(frozen=True)
class EvaluationSummary:
    generated_structures: int
    generated_sequences: int
    predicted_structures: int
    worker_count: int
    representative_device: str | None
    representative_generated_sequences: int
    representative_predicted_structures: int
    inverse_fold_model_init_seconds: float
    inverse_fold_seconds: float
    fold_model_init_seconds: float
    fold_seconds: float
    inverse_fold_model_init_wall_seconds: float
    inverse_fold_wall_seconds: float
    fold_model_init_wall_seconds: float
    fold_wall_seconds: float
    fold_model_name: str
    fold_mode: str
    fold_num_models: int | None
    fold_num_recycles: int | None
    success_summaries: tuple[SuccessSummary, ...]

    @property
    def startup_seconds(self) -> float:
        return self.inverse_fold_model_init_seconds + self.fold_model_init_seconds

    @property
    def startup_wall_seconds(self) -> float:
        return self.inverse_fold_model_init_wall_seconds + self.fold_model_init_wall_seconds

    @property
    def inverse_fold_seconds_per_sequence(self) -> float | None:
        if self.generated_sequences <= 0:
            return None
        return self.inverse_fold_seconds / self.generated_sequences

    @property
    def inverse_fold_wall_seconds_per_sequence(self) -> float | None:
        denominator = (
            self.representative_generated_sequences
            if self.worker_count > 1
            else self.generated_sequences
        )
        if denominator <= 0:
            return None
        return self.inverse_fold_wall_seconds / denominator

    @property
    def fold_seconds_per_sequence(self) -> float | None:
        if self.generated_sequences <= 0:
            return None
        return self.fold_seconds / self.generated_sequences

    @property
    def fold_wall_seconds_per_sequence(self) -> float | None:
        denominator = (
            self.representative_generated_sequences
            if self.worker_count > 1
            else self.generated_sequences
        )
        if denominator <= 0:
            return None
        return self.fold_wall_seconds / denominator

    @property
    def fold_seconds_per_model_output(self) -> float | None:
        if self.predicted_structures <= 0:
            return None
        return self.fold_seconds / self.predicted_structures

    @property
    def fold_wall_seconds_per_model_output(self) -> float | None:
        denominator = (
            self.representative_predicted_structures
            if self.worker_count > 1
            else self.predicted_structures
        )
        if denominator <= 0:
            return None
        return self.fold_wall_seconds / denominator


def _merge_evaluation_summaries(summaries: list[EvaluationSummary]) -> EvaluationSummary:
    if len(summaries) == 1:
        return summaries[0]
    return EvaluationSummary(
        generated_structures=sum(s.generated_structures for s in summaries),
        generated_sequences=sum(s.generated_sequences for s in summaries),
        predicted_structures=sum(s.predicted_structures for s in summaries),
        worker_count=sum(s.worker_count for s in summaries),
        representative_device=summaries[0].representative_device,
        representative_generated_sequences=sum(s.representative_generated_sequences for s in summaries),
        representative_predicted_structures=sum(s.representative_predicted_structures for s in summaries),
        inverse_fold_model_init_seconds=sum(s.inverse_fold_model_init_seconds for s in summaries),
        inverse_fold_seconds=sum(s.inverse_fold_seconds for s in summaries),
        fold_model_init_seconds=sum(s.fold_model_init_seconds for s in summaries),
        fold_seconds=sum(s.fold_seconds for s in summaries),
        inverse_fold_model_init_wall_seconds=sum(s.inverse_fold_model_init_wall_seconds for s in summaries),
        inverse_fold_wall_seconds=sum(s.inverse_fold_wall_seconds for s in summaries),
        fold_model_init_wall_seconds=sum(s.fold_model_init_wall_seconds for s in summaries),
        fold_wall_seconds=sum(s.fold_wall_seconds for s in summaries),
        fold_model_name=summaries[0].fold_model_name,
        fold_mode=summaries[0].fold_mode,
        fold_num_models=summaries[0].fold_num_models,
        fold_num_recycles=summaries[0].fold_num_recycles,
        success_summaries=tuple(ss for s in summaries for ss in s.success_summaries),
    )


def summarize_evaluation(rootdir: str | Path, selections: list[str] | None = None) -> EvaluationSummary:
    root = Path(rootdir)
    candidate_roots = _discover_evaluation_roots(root, selections)

    generated_structures = 0
    generated_sequences = 0
    predicted_structures = 0
    worker_count = 0
    representative_device = None
    representative_generated_sequences = 0
    representative_predicted_structures = 0
    inverse_fold_model_init_seconds = 0.0
    inverse_fold_seconds = 0.0
    fold_model_init_seconds = 0.0
    fold_seconds = 0.0
    inverse_fold_model_init_wall_seconds = 0.0
    inverse_fold_wall_seconds = 0.0
    fold_model_init_wall_seconds = 0.0
    fold_wall_seconds = 0.0
    fold_model_name = "unknown"
    fold_mode = "unknown"
    fold_num_models = None
    fold_num_recycles = None
    success_by_label: dict[str, SuccessSummary] = {}

    for candidate_root in candidate_roots:
        results_dir = candidate_root / "results"
        map_stats_path = candidate_root / "evaluation_map_stats.json"

        generated_structures += len(list((candidate_root / "pdbs").glob("*.pdb"))) if (candidate_root / "pdbs").exists() else 0
        generated_sequences += _count_fasta_entries(candidate_root / "structures" / "input.fasta")
        if (candidate_root / "structures").exists():
            predicted_structures += _count_prediction_structures(candidate_root / "structures")

        if map_stats_path.exists():
            with map_stats_path.open() as file:
                payload = json.load(file)
            worker_count += int(payload.get("worker_count", 1))
            representative_device = payload.get("representative_device", representative_device)
            representative_generated_sequences += int(
                payload.get("representative_generated_sequences", 0)
            )
            representative_predicted_structures += int(
                payload.get("representative_predicted_structures", 0)
            )
            inverse_fold_model_init_seconds += float(payload.get("inverse_fold_model_init_seconds", 0.0))
            inverse_fold_seconds += float(payload.get("inverse_fold_seconds", 0.0))
            fold_model_init_seconds += float(payload.get("fold_model_init_seconds", 0.0))
            fold_seconds += float(payload.get("fold_seconds", 0.0))
            inverse_fold_model_init_wall_seconds += float(
                payload.get(
                    "inverse_fold_model_init_wall_seconds",
                    payload.get("inverse_fold_model_init_seconds", 0.0),
                )
            )
            inverse_fold_wall_seconds += float(
                payload.get(
                    "inverse_fold_wall_seconds",
                    payload.get("inverse_fold_seconds", 0.0),
                )
            )
            fold_model_init_wall_seconds += float(
                payload.get(
                    "fold_model_init_wall_seconds",
                    payload.get("fold_model_init_seconds", 0.0),
                )
            )
            fold_wall_seconds += float(
                payload.get(
                    "fold_wall_seconds",
                    payload.get("fold_seconds", 0.0),
                )
            )
            generated_sequences = generated_sequences - _count_fasta_entries(candidate_root / "structures" / "input.fasta") + int(payload.get("generated_sequences", _count_fasta_entries(candidate_root / "structures" / "input.fasta")))
            predicted_structures = predicted_structures - _count_prediction_structures(candidate_root / "structures") + int(payload.get("predicted_structures", 0))
            fold_model_name = str(payload.get("fold_model_name", fold_model_name))
            fold_mode = str(payload.get("fold_mode", fold_mode))
            fold_num_models = payload.get("fold_num_models", fold_num_models)
            fold_num_recycles = payload.get("fold_num_recycles", fold_num_recycles)

        for success in _load_success_summaries(results_dir):
            previous = success_by_label.get(success.label)
            if previous is None:
                success_by_label[success.label] = success
            else:
                success_by_label[success.label] = SuccessSummary(
                    label=success.label,
                    successes=previous.successes + success.successes,
                    unique_tm_0_5=previous.unique_tm_0_5 + success.unique_tm_0_5,
                    unique_tm_0_6=previous.unique_tm_0_6 + success.unique_tm_0_6,
                )

    return EvaluationSummary(
        generated_structures=generated_structures,
        generated_sequences=generated_sequences,
        predicted_structures=predicted_structures,
        worker_count=worker_count,
        representative_device=representative_device,
        representative_generated_sequences=representative_generated_sequences,
        representative_predicted_structures=representative_predicted_structures,
        inverse_fold_model_init_seconds=inverse_fold_model_init_seconds,
        inverse_fold_seconds=inverse_fold_seconds,
        fold_model_init_seconds=fold_model_init_seconds,
        fold_seconds=fold_seconds,
        inverse_fold_model_init_wall_seconds=inverse_fold_model_init_wall_seconds,
        inverse_fold_wall_seconds=inverse_fold_wall_seconds,
        fold_model_init_wall_seconds=fold_model_init_wall_seconds,
        fold_wall_seconds=fold_wall_seconds,
        fold_model_name=fold_model_name,
        fold_mode=fold_mode,
        fold_num_models=fold_num_models,
        fold_num_recycles=fold_num_recycles,
        success_summaries=tuple(success_by_label.values()),
    )


def _discover_evaluation_roots(root: Path, selections: list[str] | None = None) -> list[Path]:
    direct_markers = [
        root / "pdbs",
        root / "results",
        root / "evaluation_map_stats.json",
    ]
    if any(marker.exists() for marker in direct_markers):
        return [root]
    discovered = [
        path for path in root.iterdir()
        if path.is_dir() and (
            (path / "pdbs").exists() or
            (path / "results").exists() or
            (path / "evaluation_map_stats.json").exists()
        )
    ]
    if selections is not None:
        sel_set = set(selections)
        discovered = [p for p in discovered if p.name in sel_set]
    return sorted(discovered)


def _count_fasta_entries(filepath: Path) -> int:
    if not filepath.exists():
        return 0
    count = 0
    with filepath.open() as file:
        for line in file:
            if line.startswith(">"):
                count += 1
    return count


def _count_prediction_structures(structures_dir: Path) -> int:
    if not structures_dir.exists():
        return 0
    count = 0
    for filepath in structures_dir.glob("*/*"):
        if filepath.is_file() and AF_PREDICTION_FILE_RE.match(filepath.name):
            count += 1
    return count


def _count_csv_rows(filepath: Path) -> int:
    if not filepath.exists():
        return 0
    with filepath.open(newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_unique_cluster_ids(filepath: Path) -> int:
    if not filepath.exists():
        return 0
    cluster_ids: set[str] = set()
    with filepath.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cluster_id = row.get("tm_0_6_cluster_id")
            if cluster_id:
                cluster_ids.add(cluster_id)
    return len(cluster_ids)


def _count_unique_cluster_ids_for_column(filepath: Path, column: str) -> int:
    if not filepath.exists():
        return 0
    cluster_ids: set[str] = set()
    with filepath.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cluster_id = row.get(column)
            if cluster_id:
                cluster_ids.add(cluster_id)
    return len(cluster_ids)


def _load_success_summaries(results_dir: Path) -> list[SuccessSummary]:
    candidates = [
        (
            "successes",
            results_dir / "successful_generation_info.csv",
            results_dir / "successful_generations_cluster.csv",
        ),
        (
            "backbone_successes",
            results_dir / "successful_backbone_generation_info.csv",
            results_dir / "successful_backbone_generations_cluster.csv",
        ),
        (
            "allatom_successes",
            results_dir / "successful_allatom_generation_info.csv",
            results_dir / "successful_allatom_generations_cluster.csv",
        ),
        (
            "allatom_strict_successes",
            results_dir / "successful_allatom_strict_generation_info.csv",
            results_dir / "successful_allatom_strict_generations_cluster.csv",
        ),
        (
            "v0_successes",
            results_dir / "v0_success" / "success_info.csv",
            results_dir / "v0_success" / "successful_incomplex_binders_cluster.csv",
        ),
    ]

    summaries: list[SuccessSummary] = []
    for label, info_csv, cluster_csv in candidates:
        if not info_csv.exists() and not cluster_csv.exists():
            continue
        summaries.append(
            SuccessSummary(
                label=label,
                successes=_count_csv_rows(info_csv),
                unique_tm_0_5=_count_unique_cluster_ids_for_column(cluster_csv, "tm_0_5_cluster_id"),
                unique_tm_0_6=_count_unique_cluster_ids_for_column(cluster_csv, "tm_0_6_cluster_id"),
            )
        )
    return summaries
