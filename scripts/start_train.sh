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
python_dir="$(cd -- "$(dirname -- "${python_bin}")" && pwd -P)"
python_bin="${python_dir}/$(basename -- "${python_bin}")"

# Pass only this path into tmux, never the credential file's contents. The
# actual training shell sources it after tmux/login-shell environment setup.
swanlab_env_file="${FRUGALOPL_SWANLAB_ENV_FILE:-}"
if [[ -n "${swanlab_env_file}" ]]; then
    if [[ ! -f "${swanlab_env_file}" || ! -r "${swanlab_env_file}" ]]; then
        echo "Error: SwanLab environment file is not readable: ${swanlab_env_file}" >&2
        exit 1
    fi
    swanlab_env_file="$(cd -- "$(dirname -- "${swanlab_env_file}")" && pwd -P)/$(basename -- "${swanlab_env_file}")"
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
log_path="${GPU_LOG:-${log_dir}/${session_name}.log}"
if [[ "${log_path}" != /* ]]; then
    echo "Error: GPU_LOG must be an absolute path: ${log_path}" >&2
    exit 1
fi
log_path="$(realpath -m -- "${log_path}")"
case "${config_path}" in
    "${project_root}/tests/"*)
        if [[ "${log_path}" != "${project_root}/tests/outputs/"* ]]; then
            echo "Error: test training logs must stay under tests/outputs/." >&2
            exit 1
        fi
        ;;
esac
mkdir -p -- "$(dirname -- "${log_path}")"
export GPU_LOG="${log_path}"
status_path="${log_dir}/${session_name}.status"
rm -f -- "${status_path}"

# Keep compilation/model caches in this project's writable output area, not
# a shared account's potentially root-owned home cache. Explicit overrides,
# including empty values, remain authoritative.
cache_root="${log_dir%/.logs}/.cache"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR-${cache_root}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR-${cache_root}/inductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR-${cache_root}/torch_extensions}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT-${cache_root}/vllm}"
export HF_HOME="${HF_HOME-${cache_root}/huggingface}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH-${cache_root}/cuda}"
export TORCH_HOME="${TORCH_HOME-${cache_root}/torch}"
for cache_variable in TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR \
    VLLM_CACHE_ROOT HF_HOME CUDA_CACHE_PATH TORCH_HOME; do
    cache_path="${!cache_variable}"
    if [[ -n "${cache_path}" ]]; then
        mkdir -p -- "${cache_path}"
    fi
done

# An existing tmux server can have a stale environment. Put task registration
# directly on the eventual Python command rather than relying on inheritance.
# printf %q preserves values such as GPU_TASK="logic RL" as one argument.
runtime_variables=(
    CONDA_PREFIX CUDA_HOME CUDA_PATH LD_LIBRARY_PATH TORCH_EXTENSIONS_DIR SWANLAB_RUN_ID
    TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR HF_HOME VLLM_CACHE_ROOT
    CUDA_CACHE_PATH TORCH_HOME FRUGALOPL_MODEL_PATH
)
training_environment=(env)
# A variable absent from the launching shell must not leak in from an older
# tmux server. Preserve explicitly supplied values, including empty strings.
for runtime_variable in "${runtime_variables[@]}"; do
    if [[ ! -v "${runtime_variable}" ]]; then
        training_environment+=(-u "${runtime_variable}")
    fi
done
training_environment+=(
    PYTHONUNBUFFERED=1
    # vLLM names EngineCore/Worker processes through setproctitle. Preserve
    # /proc/<pid>/environ so external GPU ownership checks retain all labels.
    SPT_NOENV=1
    "PATH=${python_dir}:${PATH}"
    "OMP_NUM_THREADS=${OMP_NUM_THREADS-4}"
    "MKL_NUM_THREADS=${MKL_NUM_THREADS-4}"
    "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS-4}"
    "MAX_JOBS=${MAX_JOBS-4}"
    "TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS-2}"
)
for environment_variable in GPU_OWNER GPU_TASK GPU_HOURS GPU_KIND GPU_LOG GPU_RUN "${runtime_variables[@]}"; do
    if [[ -v "${environment_variable}" ]]; then
        training_environment+=("${environment_variable}=${!environment_variable}")
    fi
done
printf -v python_command '%q ' \
    "${training_environment[@]}" \
    "${python_bin}" -m engine.controller \
    --config "${config_path}"
python_command="${python_command% }"
printf -v quoted_log '%q' "${log_path}"
printf -v quoted_status '%q' "${status_path}"
swanlab_setup=""
if [[ -n "${swanlab_env_file}" ]]; then
    printf -v quoted_swanlab_env '%q' "${swanlab_env_file}"
    # Suppress output from the private file as well as shell tracing. Its
    # assignments are exported without placing SWANLAB_API_KEY in any argv.
    swanlab_setup="set -a; if ! source ${quoted_swanlab_env} >/dev/null 2>&1; then set +x; echo 'Error: could not load SwanLab environment file.' >&2; exit 1; fi; set +x; set +a; "
fi
tmux_script="set +x; set -o pipefail; ${swanlab_setup}${python_command} 2>&1 | tee -a ${quoted_log}; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\${status}\" > ${quoted_status}; exit \"\${status}\""
printf -v quoted_tmux_script '%q' "${tmux_script}"

tmux new-session -d -s "${session_name}" -c "${project_root}" "bash -lc ${quoted_tmux_script}"

echo "Training started in tmux session: ${session_name}"
echo "Log: ${log_path}"
echo "Status file: ${status_path}"
echo "Attach with: tmux attach -t ${session_name}"
