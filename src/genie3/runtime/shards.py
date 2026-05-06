"""Shared shard bookkeeping: marker files, PDB directory setup, and selection helpers."""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def parse_selections(selections_str: str | None) -> list[str] | None:
    """Parse a comma-separated selections string into a list of stripped names."""
    if not selections_str:
        return None
    return [s.strip() for s in selections_str.split(",")]


def filter_problem_dirs(problem_dirs: list, selections: list[str] | None) -> list:
    """Return only those problem dirs whose basename is in *selections* (or all if None)."""
    if not selections:
        return problem_dirs
    sel_set = set(selections)
    return [p for p in problem_dirs if Path(p).name in sel_set]


# ---------------------------------------------------------------------------
# Problem directory discovery
# ---------------------------------------------------------------------------

def discover_problem_dirs(rootdir: str) -> list[str]:
    """Return directories under *rootdir* that contain a pdbs/ subdir."""
    root = Path(rootdir)
    if (root / "pdbs").exists():
        return [rootdir]
    return [str(d) for d in sorted(root.iterdir()) if d.is_dir() and (d / "pdbs").exists()]


# ---------------------------------------------------------------------------
# Generation shard markers
# ---------------------------------------------------------------------------

def is_generation_shard_done(
    outdir: Path,
    shard_id: int,
    num_shards: int,
    selected_problems: list[str] | None = None,
) -> bool:
    """Return True if generate_shard_i_of_n.done exists for every target dir."""
    marker_name = f"generate_shard_{shard_id}_of_{num_shards}.done"
    problem_dirs = (
        [d for d in outdir.iterdir() if d.is_dir() and (d / "pdbs").exists()]
        if outdir.exists()
        else []
    )
    if selected_problems is not None:
        problem_dirs = [d for d in problem_dirs if d.name in selected_problems]
    targets = problem_dirs if problem_dirs else [outdir]
    return all((t / ".shard_markers" / marker_name).exists() for t in targets)


def write_generation_shard_marker(
    outdir: Path,
    shard_id: int,
    num_shards: int,
    selected_problems: list[str] | None = None,
) -> None:
    """Write generate_shard_* completion markers into each target dir's .shard_markers/."""
    problem_dirs = [d for d in outdir.iterdir() if d.is_dir() and (d / "pdbs").exists()]
    if selected_problems is not None:
        problem_dirs = [d for d in problem_dirs if d.name in selected_problems]
    targets = problem_dirs if problem_dirs else [outdir]
    for target in targets:
        marker_dir = target / ".shard_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / f"generate_shard_{shard_id}_of_{num_shards}.done").touch()


def check_generation_markers(marker_dir: Path) -> None:
    """Raise if generate_shard_* markers are present but incomplete."""
    if not marker_dir.exists():
        return
    pattern = re.compile(r"^generate_shard_(\d+)_of_(\d+)\.done$")
    num_shards = None
    for f in marker_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            num_shards = int(m.group(2))
            break
    if num_shards is None:
        return
    missing = [
        i for i in range(num_shards)
        if not (marker_dir / f"generate_shard_{i}_of_{num_shards}.done").exists()
    ]
    if missing:
        raise RuntimeError(
            f"Generation is not complete in {marker_dir.parent} — missing shards: {missing}. "
            f"Run the missing generate shards before evaluating."
        )


def assert_generation_complete(
    rootdir: str,
    problem_dirs: list[str] | None = None,
) -> None:
    """Raise if generation shards are started but incomplete for any problem dir."""
    root = Path(rootdir)
    if problem_dirs is None:
        scanned = [d for d in root.iterdir() if d.is_dir() and (d / "pdbs").exists()]
    else:
        scanned = [Path(p) for p in problem_dirs]
    marker_roots = scanned if scanned else [root]
    for marker_root in marker_roots:
        check_generation_markers(marker_root / ".shard_markers")


# ---------------------------------------------------------------------------
# Evaluation shard helpers
# ---------------------------------------------------------------------------

def create_eval_shard_dir(problem_dir: Path, shard_id: int, num_shards: int) -> Path | None:
    """Create eval_shards/shard_i_of_n/pdbs/ with symlinks to this shard's PDB subset.

    Returns the shard dir path, or None if this shard has no PDBs assigned.
    """
    pdbs = sorted((problem_dir / "pdbs").glob("*.pdb"))
    if not pdbs:
        return None
    per_shard = math.ceil(len(pdbs) / num_shards)
    start = min(shard_id * per_shard, len(pdbs))
    end = min(start + per_shard, len(pdbs))
    assigned = pdbs[start:end]
    if not assigned:
        return None

    shard_dir = problem_dir / "eval_shards" / f"shard_{shard_id}_of_{num_shards}"
    pdb_dir = shard_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    for pdb in assigned:
        dst = pdb_dir / pdb.name
        if not dst.exists():
            dst.symlink_to(pdb.resolve())
    return shard_dir


def merge_eval_shard_outputs(problem_dir: Path, num_shards: int) -> None:
    """Copy sequences/ and structures/ from all shard subdirs into problem_dir.

    On re-reduce, shard-level sequences/structures may have been cleaned up after
    the previous reduce. In that case the copy loop is a no-op and the already-merged
    problem-root sequences/ and structures/ are reused as-is.
    """
    seq_dst = problem_dir / "sequences"
    str_dst = problem_dir / "structures"
    seq_dst.mkdir(exist_ok=True)
    str_dst.mkdir(exist_ok=True)

    for i in range(num_shards):
        marker = problem_dir / ".shard_markers" / f"evaluate_shard_{i}_of_{num_shards}.done"
        if not marker.exists():
            raise RuntimeError(
                f"Eval shard {i} of {num_shards} is not marked done in {problem_dir}. "
                f"Re-run: genie3 evaluate -c <config> --shard-id {i} --num-shards {num_shards}"
            )
        shard_dir = problem_dir / "eval_shards" / f"shard_{i}_of_{num_shards}"
        shard_seq = shard_dir / "sequences"
        shard_str = shard_dir / "structures"
        if shard_seq.exists():
            for f in shard_seq.glob("*"):
                shutil.copy2(f, seq_dst / f.name)
        if shard_str.exists():
            for d in shard_str.iterdir():
                dst = str_dst / d.name
                if d.is_dir():
                    if not dst.exists():
                        shutil.copytree(d, dst)
                elif d.suffix in {".fasta", ".fa", ".faa"}:
                    # Each shard has a different subset of sequences — concatenate.
                    with open(d, "rb") as src_f, open(dst, "ab") as dst_f:
                        dst_f.write(src_f.read())
                elif not dst.exists():
                    shutil.copy2(d, dst)


def write_eval_shard_marker(problem_dir: Path, shard_id: int, num_shards: int) -> None:
    marker_dir = problem_dir / ".shard_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / f"evaluate_shard_{shard_id}_of_{num_shards}.done").touch()


def detect_num_gen_shards(marker_dir: Path) -> int | None:
    """Return num_shards from generate_shard_* marker filenames, or None if not found."""
    if not marker_dir.exists():
        return None
    pattern = re.compile(r"^generate_shard_(\d+)_of_(\d+)\.done$")
    for f in marker_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            return int(m.group(2))
    return None


def shard_done_missing(
    marker_dir: Path, stage: str, num_shards: int
) -> tuple[list[int], list[int]]:
    """Return (done, missing) shard index lists for the given stage and num_shards."""
    done = [
        i for i in range(num_shards)
        if (marker_dir / f"{stage}_shard_{i}_of_{num_shards}.done").exists()
    ]
    missing = [i for i in range(num_shards) if i not in done]
    return done, missing


def detect_num_eval_shards(problem_dir: Path) -> int | None:
    """Return num_shards from evaluate_shard_* marker filenames, or None if not found."""
    marker_dir = problem_dir / ".shard_markers"
    if not marker_dir.exists():
        return None
    pattern = re.compile(r"^evaluate_shard_(\d+)_of_(\d+)\.done$")
    for f in marker_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            return int(m.group(2))
    return None


def assert_eval_shards_complete(problem_dir: Path) -> None:
    """Raise if eval shards are started but incomplete."""
    num_shards = detect_num_eval_shards(problem_dir)
    if num_shards is None:
        return
    marker_dir = problem_dir / ".shard_markers"
    missing = [
        i for i in range(num_shards)
        if not (marker_dir / f"evaluate_shard_{i}_of_{num_shards}.done").exists()
    ]
    if missing:
        raise RuntimeError(
            f"Eval shards are not complete in {problem_dir} — missing shards: {missing}. "
            f"Run the missing evaluate shards before reducing."
        )
