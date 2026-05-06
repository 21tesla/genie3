"""Generation summary helpers for the Genie3 CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationSummary:
    generated_samples: int
    batch_count: int
    worker_count: int
    representative_device: str | None
    total_batch_seconds: float
    aggregate_device_seconds: float
    compile_warmup_seconds: float
    representative_generated_samples: int
    representative_batch_count: int
    search_name: str | None = None
    beam_width: int | None = None

    @property
    def per_sample_seconds(self) -> float | None:
        if self.representative_generated_samples <= 0:
            return None
        return self.total_batch_seconds / self.representative_generated_samples

    @property
    def per_beam_search_seconds(self) -> float | None:
        if self.search_name != "beam" or self.representative_batch_count <= 0:
            return None
        return self.total_batch_seconds / self.representative_batch_count


def summarize_generation(rootdir: str | Path) -> GenerationSummary:
    root = Path(rootdir)
    stats_dir = root / "generation_stats"

    stage_order = ["main", "sidechain"]
    stage_stats: dict[str, dict[str, float | int | str | None]] = {}
    worker_names: set[int] = set()
    representative_rank: int | None = None

    if stats_dir.exists():
        for filepath in sorted(stats_dir.glob("*.json")):
            with filepath.open() as file:
                payload = json.load(file)

            stage_name = str(payload.get("stage", "main"))
            stage = stage_stats.setdefault(
                stage_name,
                {
                    "generated_samples": 0,
                    "batch_count": 0,
                    "aggregate_device_seconds": 0.0,
                    "representative_total_batch_seconds": 0.0,
                    "representative_compile_warmup_seconds": 0.0,
                    "representative_generated_samples": 0,
                    "representative_batch_count": 0,
                    "representative_rank": None,
                    "search_name": None,
                    "beam_width": None,
                },
            )

            rank = int(payload.get("rank", 0))
            batch_seconds_total = float(payload.get("batch_seconds_total", 0.0))
            sample_count = int(payload.get("sample_count", 0))
            output_count = int(payload.get("output_count", sample_count))

            stage["generated_samples"] += output_count
            stage["batch_count"] += sample_count
            stage["aggregate_device_seconds"] += batch_seconds_total

            stage_representative_rank = stage["representative_rank"]
            if stage_representative_rank is None or (
                stage_representative_rank != 0 and rank == 0
            ):
                stage["representative_total_batch_seconds"] = batch_seconds_total
                stage["representative_compile_warmup_seconds"] = float(
                    payload.get("compile_warmup_seconds", 0.0)
                )
                stage["representative_generated_samples"] = output_count
                stage["representative_batch_count"] = sample_count
                stage["representative_rank"] = rank
                stage["search_name"] = payload.get("search_name")
                stage["beam_width"] = payload.get("beam_width")
            worker_names.add(rank)

    ordered_stage_names = [
        name for name in stage_order if name in stage_stats
    ] + sorted(name for name in stage_stats if name not in stage_order)
    final_stage_name = ordered_stage_names[-1] if ordered_stage_names else None

    generated_samples = 0
    batch_count = 0
    total_batch_seconds = 0.0
    aggregate_device_seconds = 0.0
    compile_warmup_seconds = 0.0
    representative_generated_samples = 0
    representative_batch_count = 0
    search_name: str | None = None
    beam_width: int | None = None

    for stage_name in ordered_stage_names:
        stage = stage_stats[stage_name]
        total_batch_seconds += float(stage["representative_total_batch_seconds"])
        aggregate_device_seconds += float(stage["aggregate_device_seconds"])
        compile_warmup_seconds += float(stage["representative_compile_warmup_seconds"])
        if stage_name == final_stage_name:
            generated_samples = int(stage["generated_samples"])
            batch_count = int(stage["batch_count"])
            representative_generated_samples = int(
                stage["representative_generated_samples"]
            )
            representative_batch_count = int(stage["representative_batch_count"])
            search_name = (
                str(stage["search_name"]) if stage["search_name"] is not None else None
            )
            beam_width = (
                int(stage["beam_width"]) if stage["beam_width"] is not None else None
            )
            if stage["representative_rank"] is not None:
                representative_rank = int(stage["representative_rank"])

    if generated_samples == 0:
        generated_samples = len(list(root.rglob("pdbs/*.pdb")))

    worker_count = len(worker_names)
    if worker_count == 0 and generated_samples > 0:
        worker_count = 1

    return GenerationSummary(
        generated_samples=generated_samples,
        batch_count=batch_count,
        worker_count=worker_count,
        representative_device=(
            f"cuda:{representative_rank}" if representative_rank is not None else None
        ),
        total_batch_seconds=total_batch_seconds,
        aggregate_device_seconds=aggregate_device_seconds,
        compile_warmup_seconds=compile_warmup_seconds,
        representative_generated_samples=representative_generated_samples,
        representative_batch_count=representative_batch_count,
        search_name=search_name,
        beam_width=beam_width,
    )
