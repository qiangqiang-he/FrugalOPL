#!/usr/bin/env bash

# Start one Controller-managed training run in a detached tmux session.  The
# training YAML is a complete, recursively composed configuration: its
# explicit ``extends`` chain supplies the framework and hardware profile.
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
if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is required to start training durably." >&2
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
    python_bin="$(command -v python)"
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Error: Python interpreter is not executable: ${python_bin}" >&2
    exit 1
fi

config_stem="$(basename -- "${config_path}")"
config_stem="${config_stem%.*}"
config_stem="${config_stem//[^[:alnum:]_-]/-}"
default_session="frugalopl-train-${config_stem}-$(date +%Y%m%d-%H%M%S)-$$"
session_name="${FRUGALOPL_TMUX_SESSION:-${default_session}}"
session_name="${session_name//./_}"

if [[ ! "${session_name}" =~ ^[[:alnum:]_.-]+$ ]]; then
    echo "Error: FRUGALOPL_TMUX_SESSION contains unsupported characters: ${session_name}" >&2
    exit 1
fi
if tmux has-session -t "=${session_name}" 2>/dev/null; then
    echo "Error: tmux session already exists: ${session_name}" >&2
    exit 1
fi

# Test configurations are intentionally self-contained under tests/: keep
# their launcher logs/status files alongside their checkpoints and outputs.
# Formal configurations continue to use the project-level outputs directory.
case "${config_path}" in
    "${project_root}/tests/"*)
        log_dir="${project_root}/tests/outputs/.logs"
        ;;
    *)
        log_dir="${project_root}/outputs/.logs"
        ;;
esac
mkdir -p -- "${log_dir}"
log_path="${log_dir}/${session_name}.log"
status_path="${log_dir}/${session_name}.status"
rm -f -- "${status_path}"

printf -v python_command '%q ' \
    "${python_bin}" -m engine.controller \
    --config "${config_path}"
python_command="${python_command% }"
printf -v quoted_log '%q' "${log_path}"
printf -v quoted_status '%q' "${status_path}"
tmux_script="set -o pipefail; ${python_command} 2>&1 | tee -a ${quoted_log}; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\${status}\" > ${quoted_status}; exit \"\${status}\""
printf -v quoted_tmux_script '%q' "${tmux_script}"

tmux new-session -d -s "${session_name}" -c "${project_root}" "bash -lc ${quoted_tmux_script}"

echo "Training started in tmux session: ${session_name}"
echo "Log: ${log_path}"
echo "Status file: ${status_path}"
echo "Attach with: tmux attach -t ${session_name}"
