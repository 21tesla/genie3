#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERBOSE=0
LOG_DIR="$REPO_ROOT/logs/setup"
LOG_FILE=""
CURRENT_SECTION=""

section() {
    CURRENT_SECTION="$1"
    echo
    echo "[$1]"
}

status_line() {
    printf '  %-12s %s\n' "$1" "$2"
}

status_run() {
    status_line "RUN" "$1"
}

status_done() {
    status_line "DONE" "$1"
}

status_skip() {
    status_line "SKIP" "$1"
}

status_note() {
    status_line "NOTE" "$1"
}

run_logged() {
    if [ "$VERBOSE" -eq 1 ]; then
        "$@"
        return
    fi

    if ! "$@" >>"$LOG_FILE" 2>&1; then
        echo
        echo "Setup failed while running: $*" >&2
        echo "Detailed log: $LOG_FILE" >&2
        exit 1
    fi
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/setup/setup.sh [--verbose]
EOF
}

if [ $# -gt 0 ]; then
    case "$1" in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
fi

if [ $# -gt 0 ]; then
    echo "Unknown extra arguments: $*" >&2
    usage
    exit 1
fi

ENV_NAME="genie3"
CONDA_CREATE_ARGS=(python=3.10)
CONDA_PYTHON=""
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d-%H%M%S).log"

set +u
eval "$(conda shell.bash hook)"
set -u

if [ "$VERBOSE" -eq 0 ]; then
    status_note "Detailed installer output: $LOG_FILE"
fi

ensure_conda_env() {
    section "Conda environment"
    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        status_skip "Environment already exists ($ENV_NAME)"
    else
        status_run "Creating environment ($ENV_NAME)"
        run_logged conda create --name "$ENV_NAME" "${CONDA_CREATE_ARGS[@]}" -y
        status_done "Created environment ($ENV_NAME)"
    fi

    status_run "Activating environment ($ENV_NAME)"
    
    # --- FIX START ---
    set +u  # Temporarily disable 'unbound variable' check
    conda activate "$ENV_NAME"
    set -u  # Re-enable 'unbound variable' check
    # --- FIX END ---

    run_logged conda install pip git -y
    CONDA_PYTHON="$CONDA_PREFIX/bin/python"
    status_done "Using Python $CONDA_PYTHON"
}

install_genie3() {
    section "Genie3 package"
    status_run "Bootstrapping setuptools, wheel, numpy>=2.0.2, and Cython"
    run_conda_python_module pip install --upgrade setuptools wheel "numpy>=2.0.2,<3" Cython
    status_done "Bootstrap dependencies ready"
    status_run "Installing editable package"
    cd "$REPO_ROOT"
    run_conda_python_module pip install --no-build-isolation -e .
    status_done "Editable package installed"
}

ensure_packages_dir() {
    mkdir -p "$REPO_ROOT/packages"
    cd "$REPO_ROOT/packages"
}

configure_package_cache() {
    local default_root="${REPO_ROOT}/packages/.cache"
    local cache_root="${GENIE3_CACHE_ROOT:-${PSCRATCH:-${SCRATCH:-${TMPDIR:-$default_root}}}}"

    export GENIE3_CACHE_ROOT="$cache_root"
    export PIP_CACHE_DIR="$GENIE3_CACHE_ROOT/pip"
    export TMPDIR="$GENIE3_CACHE_ROOT/tmp"
    export XDG_CACHE_HOME="$GENIE3_CACHE_ROOT/xdg-cache"
    export GENIE3_COLABFOLD_DATA_DIR="$XDG_CACHE_HOME/colabfold"

    mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$GENIE3_COLABFOLD_DATA_DIR"
    status_note "Cache root: $GENIE3_CACHE_ROOT"
}

run_conda_python_module() {
    if [ "$VERBOSE" -eq 1 ]; then
        env -u PYTHONHOME -u PYTHONPATH -u PYTHONUSERBASE \
            PYTHONNOUSERSITE=1 \
            "$CONDA_PYTHON" -m "$@"
        return
    fi

    if ! env -u PYTHONHOME -u PYTHONPATH -u PYTHONUSERBASE \
        PYTHONNOUSERSITE=1 \
        "$CONDA_PYTHON" -m "$@" >>"$LOG_FILE" 2>&1; then
        echo
        echo "Setup failed while running: $CONDA_PYTHON -m $*" >&2
        echo "Detailed log: $LOG_FILE" >&2
        exit 1
    fi
}

run_conda_python_code() {
    if [ "$VERBOSE" -eq 1 ]; then
        env -u PYTHONHOME -u PYTHONPATH -u PYTHONUSERBASE \
            PYTHONNOUSERSITE=1 \
            XDG_CACHE_HOME="$XDG_CACHE_HOME" \
            GENIE3_COLABFOLD_DATA_DIR="$GENIE3_COLABFOLD_DATA_DIR" \
            "$CONDA_PYTHON" -c "$1"
        return
    fi

    if ! env -u PYTHONHOME -u PYTHONPATH -u PYTHONUSERBASE \
        PYTHONNOUSERSITE=1 \
        XDG_CACHE_HOME="$XDG_CACHE_HOME" \
        GENIE3_COLABFOLD_DATA_DIR="$GENIE3_COLABFOLD_DATA_DIR" \
        "$CONDA_PYTHON" -c "$1" >>"$LOG_FILE" 2>&1; then
        echo
        echo "Setup failed while running Python code in $CONDA_PYTHON" >&2
        echo "Detailed log: $LOG_FILE" >&2
        exit 1
    fi
}

register_activation_script() {
    local name="$1"
    local activate_body="$2"
    local deactivate_body="${3:-}"

    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    printf '%s\n' "$activate_body" > "$CONDA_PREFIX/etc/conda/activate.d/$name"

    if [ -n "$deactivate_body" ]; then
        mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
        printf '%s\n' "$deactivate_body" > "$CONDA_PREFIX/etc/conda/deactivate.d/$name"
    fi
}

remove_activation_script() {
    local name="$1"
    rm -f "$CONDA_PREFIX/etc/conda/activate.d/$name"
    rm -f "$CONDA_PREFIX/etc/conda/deactivate.d/$name"
}

install_foldseek() {
    section "FoldSeek"
    status_run "Installing FoldSeek"
    set +u
    run_logged conda install -c conda-forge -c bioconda foldseek -y
    set -u 
    status_done "FoldSeek ready"
}

install_proteinmpnn() {
    ensure_packages_dir
    section "ProteinMPNN"
    if [ ! -d ProteinMPNN ]; then
        status_run "Cloning ProteinMPNN"
        run_logged git clone https://github.com/dauparas/ProteinMPNN.git
        status_done "ProteinMPNN installed"
    else
        status_skip "ProteinMPNN already exists"
    fi
}

install_ipsae() {
    ensure_packages_dir
    section "IPSAE"
    if [ ! -d IPSAE ]; then
        status_run "Cloning IPSAE"
        run_logged git clone https://github.com/DunbrackLab/IPSAE.git
        status_done "IPSAE installed"
    else
        status_skip "IPSAE already exists"
    fi
}

install_tmscore() {
    ensure_packages_dir
    section "TMscore"
    if [ ! -d TMscore ]; then
        status_run "Building TMscore and TMalign"
        mkdir -p TMscore
        cd TMscore
        run_logged env -u LD_LIBRARY_PATH wget https://zhanggroup.org/TM-score/TMscore.cpp
        run_logged g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp
        run_logged chmod +x TMscore
        run_logged env -u LD_LIBRARY_PATH wget https://zhanggroup.org/TM-align/TMalign.cpp
        run_logged g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
        run_logged chmod +x TMalign
        status_done "TMscore tools built"
    else
        status_skip "TMscore tools already exist"
    fi
}

install_dssp() {
    ensure_packages_dir
    section "DSSP"
    if [ ! -d dssp-2.3.0 ]; then
        status_run "Installing mkdssp helper"
        mkdir -p dssp-2.3.0
        cd dssp-2.3.0
        run_logged env -u LD_LIBRARY_PATH wget https://github.com/martinpacesa/BindCraft/raw/refs/heads/main/functions/dssp
        run_logged mv dssp mkdssp
        run_logged chmod +x mkdssp
        status_done "DSSP helper installed"
    else
        status_skip "DSSP helper already exists"
    fi
}

install_colabfold() {
    section "ColabFold"
    status_run "Installing ColabFold conda prerequisites"
    set +u
    run_logged conda install -c conda-forge -c bioconda \
        kalign2=2.04 \
        hhsuite=3.3.0 \
        mmseqs2=18.8cc5c \
        openmm \
        "python=3.10" \
        -y
    set -u
    status_done "ColabFold conda prerequisites ready"

    status_run "Installing ColabFold Python packages into $ENV_NAME"
    run_conda_python_module pip install "colabfold[alphafold]"
    run_conda_python_module pip install --upgrade "jax[cuda12_pip]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    status_done "ColabFold ready"

    status_run "Downloading ColabFold multimer weights"
    run_conda_python_code \
        "import os
from inspect import signature
from pathlib import Path
from colabfold.download import download_alphafold_params

data_dir = Path(os.environ['GENIE3_COLABFOLD_DATA_DIR'])
data_dir.mkdir(parents=True, exist_ok=True)
kwargs = {}
if 'data_dir' in signature(download_alphafold_params).parameters:
    kwargs['data_dir'] = data_dir
download_alphafold_params('alphafold2_multimer_v3', **kwargs)"
    status_done "ColabFold multimer weights ready"

    remove_activation_script "colabfold_path.sh"
    register_activation_script \
        "colabfold_env.sh" \
        "export XDG_CACHE_HOME=\"$XDG_CACHE_HOME\"
export GENIE3_COLABFOLD_DATA_DIR=\"$GENIE3_COLABFOLD_DATA_DIR\"" \
        "unset XDG_CACHE_HOME
unset GENIE3_COLABFOLD_DATA_DIR"
}


install_esmfold() {
    section "ESMFold"

    status_run "Installing CUDA 12.8, GCC 12, and Ninja"
    set +u
    run_logged conda install -y \
        -c nvidia \
        -c conda-forge \
        cuda-toolkit=12.8 \
        gcc_linux-64=12 \
        gxx_linux-64=12 \
        ninja
    set -u

    local CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    local CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
    local CUDA_HOME="$CONDA_PREFIX"

    register_activation_script \
        "esmfold_env.sh" \
        "export LIBRARY_PATH=\"$CONDA_PREFIX/lib:\${LIBRARY_PATH:-}\"
export CC=\"$CC\"
export CXX=\"$CXX\"
export CUDA_HOME=\"$CUDA_HOME\"" \
        "export LIBRARY_PATH=\"\${LIBRARY_PATH#$CONDA_PREFIX/lib:}\"
unset CC
unset CXX
unset CUDA_HOME"

    status_run "Installing ESMFold Python dependencies"
    run_conda_python_module pip install omegaconf dm-tree modelcif
    run_conda_python_module pip install git+https://github.com/NVIDIA/dllogger.git

    status_run "Installing OpenFold (Targeting Blackwell arch)"
    # Temporarily set LD_LIBRARY_PATH only for the build step if needed
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    TORCH_CUDA_ARCH_LIST="10.0;9.0;8.9" \
    FORCE_CUDA="1" \
    run_conda_python_module pip install --no-build-isolation git+https://github.com/sokrypton/openfold.git
    
    status_run "Installing ESM and DeepSpeed"
    run_conda_python_module pip install git+https://github.com/sokrypton/esm.git
    
    # DeepSpeed may also need build flags if it compiles extensions
    TORCH_CUDA_ARCH_LIST="10.0;9.0;8.9" \
    FORCE_CUDA="1" \
    run_conda_python_module pip install --upgrade deepspeed
    
    status_done "ESMFold dependencies ready"
}


ensure_conda_env
configure_package_cache
install_genie3
install_esmfold
install_colabfold
install_foldseek
install_ipsae
install_proteinmpnn
install_tmscore
install_dssp

section "Done"
status_done "Setup completed"
status_note "Activate with: conda activate $ENV_NAME"
if [ "$VERBOSE" -eq 0 ]; then
    status_note "Detailed log: $LOG_FILE"
fi
