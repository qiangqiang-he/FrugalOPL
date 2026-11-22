#!/usr/bin/env bash

# Offline evaluation: tmux by default for the dual 4090; --no-tmux directly
# executes foreground Python for the user-operated eight-GPU server.
set -euo pipefail

usage() {
    echo "Usage: bash scripts/start_eval.sh [--tmux|--no-tmux] <config.yaml>" >&2
}

launch_mode=tmux
case "${1:-}" in
    --tmux) shift ;;
    --no-tmux) launch_mode=foreground; shift ;;
    --help|-h) usage; exit 0 ;;
esac

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

if [[ "${launch_mode}" == tmux ]] && ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is required to start evaluations durably." >&2
    exit 1
fi

# Prefer the explicit override, then the installed post_train runtime.
if [[ -n "${FRUGALOPL_PYTHON:-}" ]]; then
    python_bin="${FRUGALOPL_PYTHON}"
elif [[ -x "${HOME}/miniconda3/envs/post_train/bin/python" ]]; then
    python_bin="${HOME}/miniconda3/envs/post_train/bin/python"
elif [[ -x "/usr/local/miniconda3/envs/post_train/bin/python" ]]; then
    python_bin="/usr/local/miniconda3/envs/post_train/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    python_bin="${CONDA_PREFIX}/bin/python"
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
if [[ "${launch_mode}" == tmux ]] && tmux has-session -t "=${session_name}" 2>/dev/null; then
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
log_path="${GPU_LOG:-${log_dir}/${session_name}.log}"
if [[ "${log_path}" != /* ]]; then
    echo "Error: GPU_LOG must be an absolute path: ${log_path}" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "${log_path}")"
export GPU_LOG="${log_path}"
status_path="${log_dir}/${session_name}.status"
rm -f -- "${status_path}"

# Quote all paths before passing them through tmux's shell-command interface.
# An existing tmux server may retain another task's environment. Put the GPU
# registration and selected runtime directly on the eventual Python command.
python_dir="$(dirname -- "${python_bin}")"
runtime_variables=(CONDA_PREFIX CUDA_HOME CUDA_PATH LD_LIBRARY_PATH
    TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR
    VLLM_CACHE_ROOT HF_HOME CUDA_CACHE_PATH TORCH_HOME CUDA_VISIBLE_DEVICES)
evaluation_environment=(env)
for runtime_variable in "${runtime_variables[@]}"; do
    if [[ ! -v "${runtime_variable}" ]]; then
        evaluation_environment+=(-u "${runtime_variable}")
    fi
done
evaluation_environment+=(PYTHONUNBUFFERED=1 SPT_NOENV=1 "PATH=${python_dir}:${PATH}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS-4}" "MKL_NUM_THREADS=${MKL_NUM_THREADS-4}"
    "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS-4}"
    "MAX_JOBS=${MAX_JOBS-4}" "TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS-2}")
for environment_variable in GPU_OWNER GPU_TASK GPU_HOURS GPU_KIND GPU_LOG GPU_RUN "${runtime_variables[@]}"; do
    if [[ -v "${environment_variable}" ]]; then
        evaluation_environment+=("${environment_variable}=${!environment_variable}")
    fi
done
if [[ "${launch_mode}" == foreground ]]; then
    echo "Evaluation running in foreground (no tmux)."
    echo "Log: ${log_path}"
    cd -- "${project_root}"
    exec > >(tee -a "${log_path}") 2>&1
    exec "${evaluation_environment[@]}" "${python_bin}" "${project_root}/utils/eval.py" --config "${config_path}"
fi
printf -v python_command '%q ' "${evaluation_environment[@]}" "${python_bin}" "${project_root}/utils/eval.py" --config "${config_path}"
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
