#!/usr/bin/env bash

# Start one Controller-managed training run in the foreground.
set -euo pipefail

usage() {
    echo "Usage: bash scripts/start_train.sh <algorithm-config.yaml>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
config_path="$1"

if [[ "${config_path}" != /* ]]; then
    config_path="${project_root}/${config_path}"
fi
config_path="$(cd -- "$(dirname -- "${config_path}")" && pwd -P)/$(basename -- "${config_path}")"

if [[ ! -f "${config_path}" || ! -r "${config_path}" ]]; then
    echo "Error: training configuration is not readable: ${config_path}" >&2
    exit 1
fi

# Prefer an explicit interpreter, then the active Conda environment, then the
# documented local WSL runtime.  A normal PATH fallback keeps server use
# portable without making a new environment part of the launcher.
if [[ -n "${FRUGALOPL_PYTHON:-}" ]]; then
    python_bin="${FRUGALOPL_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    python_bin="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/qqh/miniconda3/envs/py312-th210-vl019-cu128/bin/python" ]]; then
    python_bin="/home/qqh/miniconda3/envs/py312-th210-vl019-cu128/bin/python"
else
    python_bin="$(command -v python || true)"
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Error: Python interpreter is not executable: ${python_bin}" >&2
    exit 1
fi

python_dir="$(cd -- "$(dirname -- "${python_bin}")" && pwd -P)"
python_bin="${python_dir}/$(basename -- "${python_bin}")"
cd -- "${project_root}"
export PATH="${python_dir}:${PATH}"
# Keep GPU registration visible after vLLM renames its worker processes.
export SPT_NOENV=1

if [[ -n "${GPU_LOG:-}" ]]; then
    if [[ "${GPU_LOG}" != /* ]]; then
        echo "Error: GPU_LOG must be an absolute path: ${GPU_LOG}" >&2
        exit 1
    fi
    mkdir -p -- "$(dirname -- "${GPU_LOG}")"
    echo "Log: ${GPU_LOG}"
    exec >>"${GPU_LOG}" 2>&1
fi

exec "${python_bin}" -u -m engine.controller --config "${config_path}"
