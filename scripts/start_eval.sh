#!/usr/bin/env bash

# Start one durable offline evaluation.  The launcher deliberately creates a
# detached tmux session: an evaluation may run for hours, and must survive a
# terminal, SSH, or Codex-session disconnect.
set -euo pipefail

usage() {
    echo "Usage: bash scripts/start_eval.sh <config.yaml>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
config_path="$1"

# Treat relative configuration paths as project-root-relative, not relative to
# whichever directory the caller happened to be in.
if [[ "${config_path}" != /* ]]; then
    config_path="${project_root}/${config_path}"
fi
config_path="$(cd -- "$(dirname -- "${config_path}")" && pwd -P)/$(basename -- "${config_path}")"

if [[ ! -f "${config_path}" ]]; then
    echo "Error: evaluation configuration was not found: ${config_path}" >&2
    exit 1
fi

if [[ ! -r "${config_path}" ]]; then
    echo "Error: evaluation configuration is not readable: ${config_path}" >&2
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is required to start evaluations durably." >&2
    exit 1
fi

# Prefer an explicitly selected interpreter, then an activated Conda env, then
# the project's required local WSL environment.  The fallback retains normal
# portability for a server whose Conda installation lives elsewhere.
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
default_session="frugalopl-eval-${config_stem}-$(date +%Y%m%d-%H%M%S)-$$"
session_name="${FRUGALOPL_TMUX_SESSION:-${default_session}}"
# tmux canonicalizes dots in session names to underscores.  Normalize before
# deriving both the tmux target and the companion log/status filenames, so a
# caller can reliably monitor the same name that tmux reports.
session_name="${session_name//./_}"

if [[ ! "${session_name}" =~ ^[[:alnum:]_.-]+$ ]]; then
    echo "Error: FRUGALOPL_TMUX_SESSION contains unsupported characters: ${session_name}" >&2
    exit 1
fi
if tmux has-session -t "=${session_name}" 2>/dev/null; then
    echo "Error: tmux session already exists: ${session_name}" >&2
    exit 1
fi

# A config stored under tests/ is a local validation artifact, so preserve its
# tmux log and exit status under tests/ as well.  Formal evaluation configs
# retain the normal project-level log root.
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

# Quote all paths before passing them through tmux's shell-command interface.
printf -v python_command '%q ' "${python_bin}" "${project_root}/utils/eval.py" --config "${config_path}"
python_command="${python_command% }"
printf -v quoted_log '%q' "${log_path}"
printf -v quoted_status '%q' "${status_path}"
tmux_script="set -o pipefail; ${python_command} 2>&1 | tee -a ${quoted_log}; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\${status}\" > ${quoted_status}; exit \"\${status}\""
printf -v quoted_tmux_script '%q' "${tmux_script}"

tmux new-session -d -s "${session_name}" -c "${project_root}" "bash -lc ${quoted_tmux_script}"

echo "Evaluation started in tmux session: ${session_name}"
echo "Log: ${log_path}"
echo "Status file: ${status_path}"
echo "Attach with: tmux attach -t ${session_name}"
