#!/bin/bash
#SBATCH --job-name=mlpinf_v5.1-dynamo_trtllm-disagg.llama3.1_405B
#SBATCH --nodes=18
#SBATCH --output=./slurm_llama405b_dynamo_disagg_18node_a01r_%j.txt
#SBATCH --time=2:00:00
#SBATCH --partition=
#SBATCH --account=
# TODO(submitter): update slurm cluster config ^

# NOTE: this script is being executed on (head node)
echo "Hostname IP: $(hostname -i)"
echo "Hostname: $(hostname)"
echo "SLURMD_NODENAME: ${SLURMD_NODENAME}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"

export HEAD_NODE="${SLURMD_NODENAME}"
echo "SLURM_HEAD_NODE: ${HEAD_NODE}"

PLATFORM="arm64"
DYNAMO_COMMIT="f9b1757f32d0793abbb625ed4467eab133924671"
TRTLLM_COMMIT="cd2f9fa62f1c7d9fa35c23518092d45d022f226c"
IMAGE="mlpinf_dynamo_trtllm:${DYNAMO_COMMIT}-${TRTLLM_COMMIT}-${PLATFORM}"

# TODO(submitter): update these
LOCAL_DYNAMO_IMAGE="/your/path/to/${IMAGE}.sqsh" # local path where we store / load image from
MODEL_PATH="/your/path/to/model/llama3.1-405b-instruct-hf-torch-fp4"
export REPO_ROOT="/your/path/to/mlperf-inference"
export LOADGEN_WHEEL="/your/path/to/mlcommons_loadgen-5.0.25-cp312-cp312-linux_aarch64.whl"
export MLPERF_SCRATCH_PATH="/your/path/to/mlperf_inference_storage"
export MLPERF_CONTAINER_IMAGE=/your/path/to/mlperf-inference.sqsh
export HF_TOKEN="<your-hf-token>"

### Dynamo Repository Setup

ROOT=$(mktemp -d -p ${PWD})
DYNAMO_ROOT="${ROOT}/dynamo"

rm -r "${DYNAMO_ROOT}"
git clone https://github.com/ai-dynamo/dynamo.git "${DYNAMO_ROOT}"
git checkout "${DYNAMO_COMMIT}"

## Dynamo Engine Configs

DYNAMO_TRTLLM_DIR="${DYNAMO_ROOT}/components/backends/trtllm"
ENGINE_CONFIG_DIR="engine_configs/llama3.1_405b"
mkdir -p "${DYNAMO_TRTLLM_DIR}/${ENGINE_CONFIG_DIR}"

# Relative paths for mounting in container
REL_PREFILL_ENGINE_CONFIG_FILE="${ENGINE_CONFIG_DIR}/prefill_nvl72.yaml"
REL_DECODE_ENGINE_CONFIG_FILE="${ENGINE_CONFIG_DIR}/decode_nvl72.yaml"

# Full paths for writing to file outside container
FULL_PREFILL_ENGINE_CONFIG_FILE="${DYNAMO_TRTLLM_DIR}/${REL_PREFILL_ENGINE_CONFIG_FILE}"
FULL_DECODE_ENGINE_CONFIG_FILE="${DYNAMO_TRTLLM_DIR}/${REL_DECODE_ENGINE_CONFIG_FILE}"

cat >"${FULL_PREFILL_ENGINE_CONFIG_FILE}" <<EOF
backend: pytorch
tensor_parallel_size: 4
pipeline_parallel_size: 1

max_batch_size: 128
max_num_tokens: 4096
# Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
max_seq_len: 20192

kv_cache_dtype: fp8
disable_overlap_scheduler: true
enable_chunked_prefill: true

kv_cache_config:
  free_gpu_memory_fraction: 0.95
  # Disable kv cache block reuse for consistent benchmarking purposes
  # but in a real service you may want this enabled.
  enable_block_reuse: false

cache_transceiver_config:
  # Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
  max_num_tokens: 20192

scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
EOF

cat >"${FULL_DECODE_ENGINE_CONFIG_FILE}" <<EOF
backend: pytorch
tensor_parallel_size: 4
pipeline_parallel_size: 1

max_batch_size: 512
max_num_tokens: 512
# Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
max_seq_len: 20192

kv_cache_dtype: fp8
disable_overlap_scheduler: false

use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64
  - 96
  - 128
  - 150
  - 180
  - 192
  - 224
  - 256
  - 288
  - 352
  - 416
  - 480
  - 512

kv_cache_config:
  free_gpu_memory_fraction: 0.95
  # Disable kv cache block reuse for consistent benchmarking purposes
  # but in a real service you may want this enabled.
  enable_block_reuse: false

cache_transceiver_config:
  # Dataset peaks at 20010 tokens, and need a multiple of block size (32) here
  max_num_tokens: 20192
EOF

## Dynamo deployment using srun_disaggregated.sh

export IMAGE=$IMAGE
export LOCAL_IMAGE=$LOCAL_DYNAMO_IMAGE
export MODEL_PATH=$MODEL_PATH
export MOUNTS="${DYNAMO_TRTLLM_DIR}:/mnt,${DYNAMO_ROOT}/benchmarks:/benchmarks"
export SERVED_MODEL_NAME="meta-llama/Llama-3.1-405B-Instruct"

export PREFILL_ENGINE_CONFIG="/mnt/${REL_PREFILL_ENGINE_CONFIG_FILE}"
export DECODE_ENGINE_CONFIG="/mnt/${REL_DECODE_ENGINE_CONFIG_FILE}"

# 14P / 4D for NVL72 (double 7P2D config for NVL36)
export NUM_GPUS_PER_NODE=4
export NUM_DECODE_NODES=1
export NUM_DECODE_WORKERS=4
export NUM_PREFILL_NODES=1
export NUM_PREFILL_WORKERS=14
export DISAGGREGATION_STRATEGY="prefill_first"

# Launch frontend, prefill, and decode workers
cd ${DYNAMO_ROOT}/components/backends/trtllm/multinode
./srun_disaggregated.sh

# Await startup
sleep_time=120 # 2 minutes
echo "$(date '+%F %T')  Waiting ${sleep_time}s for workers/server to warm-up…"
sleep "${sleep_time}"
echo "$(date '+%F %T')  Starting benchmark now"

### Mlperf inference

export ENV="dev"
export BUILD_CONTEXT="aarch64-Grace"
export CUDA_VER="12.9"
export TRT_VER="10.11.0.33"
export MITTEN_HASH="82a930d962ce6bd8aed38cf185a2acfbbfd6b84b"
export INSTALL_RGAT_DEPS="0"
export INSTALL_TRTLLM=""
export INSTALL_LOADGEN=""
export PYTHONPATH=/work/.llm_aarch64/lib/python3.12/site-packages/:${PYTHONPATH}
export MLPERF_CONTAINER_NAME=disaggr-test
export CONTAINER_MOUNTS="${REPO_ROOT}:/mlperf-inference/,${REPO_ROOT}/closed/NVIDIA/:/work,${MLPERF_SCRATCH_PATH}:/home/mlperf_inference_storage"
export WORK_DIR="${REPO_ROOT}/closed/NVIDIA/code/llama3_1-405b/repro/run_disagg_405B"

LOG_DIR="${PWD}/logs"
mkdir -p ${LOG_DIR}
echo "Using LOG_DIR: ${LOG_DIR}"

# Use the head node picked from sbatch allocation as frontend node, client node, etc.
hostname_value=$(hostname)
echo "server host name: $hostname_value"

export TRTLLM_ENDPOINT_URL="${hostname_value}:8000"
echo "TRTLLM_ENDPOINT_URL: ${TRTLLM_ENDPOINT_URL}"

echo "Starting loadgen container..."
# start the mlperf_loadgen container
srun -l --container-image=${MLPERF_CONTAINER_IMAGE} \
	--container-name=${MLPERF_CONTAINER_NAME} \
	--container-mounts=${CONTAINER_MOUNTS} \
	--nodelist="${HEAD_NODE}" \
	--overlap \
	--mpi=pmix \
	-N 1 -n 1 \
	echo "Container up."

timestamp=$(date +"%Y.%m.%d_%H.%M")
PERF_LOG="${LOG_DIR}/mlperf_benchmark_performance_${timestamp}.log"
ACC_LOG="${LOG_DIR}/mlperf_benchmark_accuracy_${timestamp}.log"

# Enter existing mlperf_loadgen container to run benchmarks

# Enable DYNAMO WARs in mlperf-inference v5.1 harness
export MLPINF_USE_DYNAMO=1

# 1. PerformanceOnly test
echo "Running PerformanceOnly benchmark, see [${PERF_LOG}]"
srun -l \
	--container-name="${MLPERF_CONTAINER_NAME}" \
	--container-mounts="${CONTAINER_MOUNTS}" \
	--container-workdir=/work \
	--container-env=HF_TOKEN,MLPINF_USE_DYNAMO \
	--overlap \
	-N 1 -n 1 \
	--mpi=pmix \
	--nodelist="${HEAD_NODE}" \
	bash -c 'pip install ${LOADGEN_WHEEL} && pip install openai orjson tokenizers transformers && SYSTEM_NAME=GB200-NVL72_GB200-186GB_aarch64x72_Dynamo make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=interactive --core_type=trtllm_endpoint --trtllm_server_urls=${TRTLLM_ENDPOINT_URL} --test_mode=PerformanceOnly --config_dir=/mlperf-inference/open/NVIDIA"' |
	tee "${PERF_LOG}" 2>&1

# 2. AccuracyOnly test
echo "Running AccuracyOnly benchmark, see [${LOG_DIR}/mlperf_benchmark_accuracy.log]"
srun -l \
	--container-name="${MLPERF_CONTAINER_NAME}" \
	--container-mounts="${CONTAINER_MOUNTS}" \
	--container-workdir=/work \
	--container-env=HF_TOKEN,MLPINF_USE_DYNAMO \
	--overlap \
	-N 1 -n 1 \
	--mpi=pmix \
	--nodelist="${HEAD_NODE}" \
	bash -c 'pip install ${LOADGEN_WHEEL} && pip install openai orjson tokenizers transformers && SYSTEM_NAME=GB200-NVL72_GB200-186GB_aarch64x72_Dynamo make run_harness RUN_ARGS="--benchmarks=llama3.1-405b --scenarios=interactive --core_type=trtllm_endpoint --trtllm_server_urls=${TRTLLM_ENDPOINT_URL} --test_mode=AccuracyOnly --config_dir=/mlperf-inference/open/NVIDIA"' |
	tee "${ACC_LOG}" 2>&1

wait

# 3. Stage results for submission: `make stage_results`

# 4. Run audit test: `make run_audit_harness ...`

# 5. Submit results

