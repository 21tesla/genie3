"""CLI-facing evaluation workflows for Genie3."""

from __future__ import annotations

import logging
import os
import shutil
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from genie3.config import to_evaluation_kwargs, validate_config_for_command
from genie3.evaluation.pipeline import Runner
from genie3.evaluation.stats import EvaluationSummary, summarize_evaluation
from genie3.runtime.context import RunContext
from genie3.runtime.progress import get_active_reporter
from genie3.runtime.shards import (
    assert_eval_shards_complete,
    assert_generation_complete,
    create_eval_shard_dir,
    detect_num_eval_shards,
    discover_problem_dirs,
    filter_problem_dirs,
    merge_eval_shard_outputs,
    parse_selections,
    write_eval_shard_marker,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Minimal evaluation result for CLI summaries."""

    output_dir: Path | None = None
    summary: EvaluationSummary | None = None
    is_reduce: bool = False


def run_evaluation(
    *,
    config_path: str,
    verbose: bool = False,
    log_dir: str | None = None,
    input_dir: Path | str | None = None,
    context: RunContext | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    reduce: bool = False,
    num_devices: int | None = None,
) -> EvaluationResult:
    """Run evaluation through the new CLI path."""
    del log_dir
    if context is not None:
        context.logger.debug("Preparing evaluation config")
        run_config = context.config
    else:
        from genie3.config import load_experiment_config

        run_config = load_experiment_config(config_path)

    if input_dir is None:
        validate_config_for_command(run_config, "evaluate")
    kwargs = to_evaluation_kwargs(run_config, input_dir=input_dir)
    if num_devices is not None:
        kwargs["n_device"] = num_devices
    kwargs["verbose"] = context.invocation.verbose if context is not None else verbose
    kwargs["shard_id"] = shard_id
    kwargs["num_shards"] = num_shards
    kwargs["reduce"] = reduce

    args = Namespace(**kwargs)

    if context is not None:
        context.logger.debug("Evaluation rootdir: %s", kwargs["rootdir"])
        context.logger.debug("Evaluation runtime args: %s", args)
        get_active_reporter().set_status("initialization")
        with context.profile.stage("evaluate"):
            main(args)
        get_active_reporter().set_status("done")
    else:
        main(args)

    rootdir = kwargs["rootdir"]
    if reduce:
        return EvaluationResult(
            output_dir=Path(rootdir) / "results",
            summary=summarize_evaluation(rootdir, selections=parse_selections(kwargs.get("selections"))),
            is_reduce=True,
        )
    else:
        from genie3.evaluation.stats import _merge_evaluation_summaries
        _all_dirs = filter_problem_dirs(
            discover_problem_dirs(rootdir),
            parse_selections(kwargs.get("selections")),
        )
        shard_summaries = [
            summarize_evaluation(str(Path(pd) / "eval_shards"))
            for pd in _all_dirs
            if (Path(pd) / "eval_shards").exists()
        ]
        summary = _merge_evaluation_summaries(shard_summaries) if shard_summaries else None
        return EvaluationResult(
            output_dir=Path(rootdir) / "results",
            summary=summary,
            is_reduce=False,
        )


def main(args: Namespace) -> None:
    """
    Run evaluation in shard mode or reduce mode.

    Shard mode (default): runs sanitize → split → map → aggregate on this
    shard's PDB subset inside an isolated eval_shards/shard_i_of_n/ subdir.

    Reduce mode (--reduce): merges all shard outputs and runs reduce on each
    problem directory. Run after all shards complete.
    """
    rootdir = args.rootdir
    version = args.version
    n_device = args.n_device
    verbose = args.verbose
    shard_id = getattr(args, "shard_id", 0)
    num_shards = getattr(args, "num_shards", 1)
    reduce = getattr(args, "reduce", False)

    _skip_keys = {"rootdir", "version", "n_device", "verbose", "shard_id", "num_shards", "reduce", "skip_reduce", "selections"}
    extra_kwargs = {k: v for k, v in vars(args).items() if k not in _skip_keys}

    problem_dirs = filter_problem_dirs(
        discover_problem_dirs(rootdir),
        parse_selections(getattr(args, "selections", None)),
    )

    if reduce:
        assert_generation_complete(rootdir, problem_dirs=problem_dirs)
        get_active_reporter().set_status("reduce")
        runner = Runner()
        for problem_dir_str in problem_dirs:
            problem_dir = Path(problem_dir_str)
            if (problem_dir / "results" / "eval.done").exists():
                logging.info("[Runner] Already reduced %s; skipping.", problem_dir_str)
                continue
            assert_eval_shards_complete(problem_dir)
            # sequences/ and structures/ at problem root are kept as the persistent
            # re-reduce source; only results/ needs to be cleared.
            stale_dir = problem_dir / "results"
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
            
            detected_num_shards = detect_num_eval_shards(problem_dir)
            if detected_num_shards is None:
                logging.warning(
                    "No eval shard markers found in %s; skipping reduce.", problem_dir_str
                )
                continue
            
            # Ensure we have something to merge before potentially clearing root.
            has_shard_data = False
            for i in range(detected_num_shards):
                shard_dir = problem_dir / "eval_shards" / f"shard_{i}_of_{detected_num_shards}"
                if (shard_dir / "sequences").exists() or (shard_dir / "structures").exists():
                    has_shard_data = True
                    break
            
            if has_shard_data:
                # If we are about to merge from shards, clear the destination first
                # to avoid mixing with stale data or duplicating sequences.
                for d in (problem_dir / "sequences", problem_dir / "structures"):
                    if d.exists():
                        shutil.rmtree(d)
                merge_eval_shard_outputs(problem_dir, detected_num_shards)
            
            runner.reduce(version, problem_dir_str, verbose, extra_kwargs)
            (problem_dir / "results").mkdir(exist_ok=True)
            (problem_dir / "results" / "eval.done").touch()
            # Clean up large shard-level outputs now merged into problem root.
            for i in range(detected_num_shards):
                shard_dir = problem_dir / "eval_shards" / f"shard_{i}_of_{detected_num_shards}"
                for subdir in ("sequences", "structures", "pdbs"):
                    d = shard_dir / subdir
                    if d.exists():
                        shutil.rmtree(d)
        return

    # Shard mode
    assert_generation_complete(rootdir, problem_dirs=problem_dirs)
    runner = Runner()
    for problem_dir_str in problem_dirs:
        problem_dir = Path(problem_dir_str)
        marker = problem_dir / ".shard_markers" / f"evaluate_shard_{shard_id}_of_{num_shards}.done"
        if marker.exists():
            logging.info(
                "[Runner] Shard %d/%d already done for %s; skipping.",
                shard_id, num_shards, problem_dir_str,
            )
            continue
        shard_dir = create_eval_shard_dir(problem_dir, shard_id, num_shards)
        if shard_dir is None:
            logging.warning(
                "[Runner] Shard %d/%d has no PDBs in %s; skipping.",
                shard_id, num_shards, problem_dir_str,
            )
        else:
            logging.info(
                "[Runner] Shard %d/%d — evaluating %s",
                shard_id, num_shards, shard_dir,
            )
            runner.evaluate(
                version=version,
                rootdir=str(shard_dir),
                n_device=n_device,
                verbose=verbose,
                skip_reduce=True,
                **extra_kwargs,
            )
        write_eval_shard_marker(problem_dir, shard_id, num_shards)
